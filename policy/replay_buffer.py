"""Replay buffer: a fixed-size ring of past transitions, kept as torch tensors on the learner's device.

SAC is off-policy: it learns from transitions collected by older versions of the policy. Every env step adds
num_envs transitions at once; when the buffer is full the oldest are overwritten.
"""

from typing import NamedTuple

import torch


class Batch(NamedTuple):
    obs: torch.Tensor       # (B, obs_dim) raw (unnormalized) observation
    action: torch.Tensor    # (B, act_dim) in [-1, 1]
    reward: torch.Tensor    # (B,)
    next_obs: torch.Tensor  # (B, obs_dim) observation right after the action (env's final_observation)
    terminal: torch.Tensor  # (B,) 1 = the episode really ended (failure): don't bootstrap from next_obs


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str | torch.device):
        self.capacity, self.device = capacity, torch.device(device)
        self.obs = torch.zeros(capacity, obs_dim, device=self.device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=self.device)
        self.action = torch.zeros(capacity, act_dim, device=self.device)
        self.reward = torch.zeros(capacity, device=self.device)
        self.terminal = torch.zeros(capacity, device=self.device)
        self.pos = 0   # next row to write
        self.size = 0  # rows filled so far

    def add(self, obs, action, reward, next_obs, terminal) -> None:
        """Add one env step: each argument has num_envs rows (any device; moved here)."""
        n = obs.shape[0]
        idx = (self.pos + torch.arange(n, device=self.device)) % self.capacity
        for buf, value in ((self.obs, obs), (self.action, action), (self.reward, reward),
                           (self.next_obs, next_obs), (self.terminal, terminal)):
            buf[idx] = value.to(self.device, torch.float32)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Uniformly random transitions (with replacement)."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return Batch(self.obs[idx], self.action[idx], self.reward[idx], self.next_obs[idx], self.terminal[idx])

    def __len__(self) -> int:
        return self.size
