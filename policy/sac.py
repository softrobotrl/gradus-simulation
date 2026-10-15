"""Soft actor-critic (Haarnoja et al. 2018, with automatic entropy tuning), in plain PyTorch.

Pieces:
  ObsNormalizer  running mean/std of observations, so inputs in mm and in rad/s reach the networks at similar scale
  Actor          observation -> squashed Gaussian over actions in [-1, 1] (the policy)
  Critic         (observation, action) -> two independent Q estimates (expected discounted future reward)
  SAC            owns all of the above plus target critics and the entropy temperature alpha; update() is one
                 gradient step on a batch from the replay buffer

The plain-words explanation of every line is in policy/README.md, "How SAC works here".
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.replay_buffer import Batch

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # keeps the policy's spread between ~0.007 and ~7 before tanh


@dataclass
class SACHyperparams:
    gamma: float = 0.99          # discount: how much future reward counts (0.99 ~ 100 steps = 2 s ahead)
    tau: float = 0.005           # target critic tracking speed (Polyak averaging)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.1      # starting entropy temperature (tuned automatically from there)
    hidden: int = 256            # width of every hidden layer
    layers: int = 2              # hidden layers per network
    grad_clip: float = 10.0      # max gradient norm per network (guards against rare huge updates)


def mlp(inp: int, out: int, hidden: int, layers: int) -> nn.Sequential:
    dims = [inp] + [hidden] * layers
    blocks = []
    for a, b in zip(dims[:-1], dims[1:]):
        blocks += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*blocks, nn.Linear(dims[-1], out))


class ObsNormalizer(nn.Module):
    """Running mean and variance over every observation seen so far (Chan et al. parallel update).

    Raw observations go in the replay buffer; they are normalized when used, with the current statistics.
    Saved in checkpoints (as buffers), because a trained policy only works with the statistics it learned with.
    """

    def __init__(self, size: int, clip: float = 5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        batch_mean, batch_var = x.mean(0), x.var(0, unbiased=False)
        delta, total = batch_mean - self.mean, self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta.square() * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, layers: int):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden, layers)  # outputs mean and log-std per action

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action in [-1, 1], log-probability of that action)."""
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        if deterministic:
            return torch.tanh(mean), torch.zeros(obs.shape[0], device=obs.device)
        std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        u = mean + std * torch.randn_like(mean)  # reparameterized sample: gradients flow through mean and std
        action = torch.tanh(u)
        # log N(u) minus log |d tanh(u)/du|, written in a numerically stable form.
        log_prob = (-0.5 * ((u - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob -= (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    """Twin Q-networks. Using the smaller of the two counters Q-learning's habit of overestimating."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, layers: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden, layers)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden, layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, hp: SACHyperparams, device: str | torch.device):
        self.hp, self.device = hp, torch.device(device)
        self.normalizer = ObsNormalizer(obs_dim).to(self.device)
        self.actor = Actor(obs_dim, act_dim, hp.hidden, hp.layers).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hp.hidden, hp.layers).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(hp.init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = -float(act_dim)  # standard choice: -1 per action dimension
        fused = self.device.type == "cuda"  # one GPU kernel per optimizer step instead of several: ~15% faster updates
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr, fused=fused)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr, fused=fused)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp.alpha_lr, fused=fused)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Actions for raw observations (any device; returned on the learner device)."""
        action, _ = self.actor(self.normalizer(obs.to(self.device)), deterministic)
        return action

    def update(self, batch: Batch) -> dict[str, float]:
        """One gradient step for the critics, the actor and alpha, then nudge the target critics."""
        obs, next_obs = self.normalizer(batch.obs), self.normalizer(batch.next_obs)
        alpha = self.alpha.detach()

        # Critic: regress both Qs onto the soft Bellman target r + gamma * (min Q_target(s', a') - alpha log pi(a'|s')).
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            soft_value = torch.min(q1_next, q2_next) - alpha * next_log_prob
            target = batch.reward + self.hp.gamma * (1 - batch.terminal) * soft_value
        q1, q2 = self.critic(obs, batch.action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self._step(self.critic_opt, critic_loss, self.critic)

        # Actor: pick actions the critic rates highly, while staying random (high entropy) where it can.
        self.critic.requires_grad_(False)  # the actor loss must not change the critic
        action, log_prob = self.actor(obs)
        q_pi = torch.min(*self.critic(obs, action))
        actor_loss = (alpha * log_prob - q_pi).mean()
        self._step(self.actor_opt, actor_loss, self.actor)
        self.critic.requires_grad_(True)

        # Alpha: raise it when the policy's entropy is below target (too deterministic), lower it when above.
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # Target critics trail the critics slowly, which keeps the Bellman target from chasing itself.
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.lerp_(p, self.hp.tau)

        return {"loss/critic": critic_loss.detach(), "loss/actor": actor_loss.detach(),
                "loss/alpha": alpha_loss.detach(), "sac/alpha": alpha, "sac/entropy": -log_prob.detach().mean(),
                "sac/q": q1.detach().mean(), "sac/target_q": target.mean()}

    def _step(self, opt: torch.optim.Optimizer, loss: torch.Tensor, net: nn.Module) -> None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), self.hp.grad_clip)
        opt.step()

    def state_dict(self) -> dict:
        return {"normalizer": self.normalizer.state_dict(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(), "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(), "alpha_opt": self.alpha_opt.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.normalizer.load_state_dict(state["normalizer"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
