"""Clone the scripted walk into a policy, and keep PPO fine-tuning close to it.

PPO from scratch has not found walking on this rig (see REWARD.md), so we start
from a policy that already walks: a behaviour clone of the scripted side-swing
gait (reward_check/gait.py), trained on the same observations the policy uses
(the 328 hardware values by default).  Then PPO fine-tunes it with a KL penalty
that keeps it close to the clone, so it cannot forget the gait in the first
few dozen iterations (which plain fine-tuning did every time):

    loss = PPO loss + beta * KL(clone || policy),  beta halving every --kl-half-life iterations, never below --kl-floor

Steps:

    python -m single_leg_demo.clone collect --out logs/clone/data.npz --dr --rail-friction 3
    python -m single_leg_demo.clone fit --out logs/single_leg_demo/clone logs/clone/data.npz
    python -m single_leg_demo.train --resume logs/single_leg_demo/clone/model_0.pt \\
        --run-name ft_kl --dr --rail-friction 3 --cmd-min 0.05 --cmd-max 0.08 \\
        --actor-freeze 30 --kl-clone 1.0 --kl-half-life 300 --warmup 0 --iterations 500

collect: the leg runs the scripted gait with a little action noise (so the data
covers slightly-off states too); the label is always the clean scripted action.
fit: supervised regression of the actor's mean action on those labels; the
actor's and critic's input normalizers are set from the data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from single_leg_demo import rewards
from single_leg_demo.reward_check.gait import ScriptedGait
from single_leg_demo.rig_env import RigConfig, SingleLegRigEnv


# --------------------------------------------------------------------------- #
# KL anchor to the clone during PPO
# --------------------------------------------------------------------------- #
class CloneKL:
    """Extra PPO loss: beta * KL(clone || policy), averaged over the minibatch.

    Plugs into rsl-rl's extra-loss slot (``PPO.symmetry``), which adds
    ``mirror_loss_coeff * compute_loss(...)`` to the PPO loss on every
    minibatch and logs it as ``Loss/symmetry``; here that value is the KL.
    beta halves every ``half_life`` PPO iterations, down to ``floor``.
    """

    use_mirror_loss = True

    def __init__(self, clone, beta: float, half_life: float, updates_per_iteration: int,
                 floor: float = 0.0):
        self.clone = clone
        self.beta0 = float(beta)
        self.floor = float(floor)
        self.half_life = float(half_life)
        self.per_it = max(int(updates_per_iteration), 1)
        self.calls = 0

    @property
    def mirror_loss_coeff(self) -> float:
        it = self.calls / self.per_it
        beta = self.beta0 * 0.5 ** (it / self.half_life) if self.half_life > 0 else self.beta0
        return max(beta, self.floor)

    def augment_batch(self, batch, original_batch_size) -> None:   # no augmentation
        pass

    def compute_loss(self, actor, batch, original_batch_size) -> torch.Tensor:
        self.calls += 1
        with torch.no_grad():
            self.clone(batch.observations, stochastic_output=True)
            ref = tuple(p[:original_batch_size] for p in self.clone.output_distribution_params)
        cur = tuple(p[:original_batch_size] for p in actor.output_distribution_params)
        return actor.get_kl_divergence(ref, cur).mean()


def install_clone_kl(alg, beta: float, half_life: float, floor: float = 0.0):
    """Freeze a copy of the current actor (the loaded clone) and anchor PPO to it."""
    import copy
    if alg.symmetry is not None:
        raise RuntimeError("rsl-rl's extra-loss slot (symmetry) is already in use")
    clone = copy.deepcopy(alg.actor).eval()
    for q in clone.parameters():
        q.requires_grad_(False)
    alg.symmetry = CloneKL(clone, beta, half_life,
                           alg.num_learning_epochs * alg.num_mini_batches, floor)
    return alg.symmetry


# --------------------------------------------------------------------------- #
# Collect and fit
# --------------------------------------------------------------------------- #
def collect_config(args) -> RigConfig:
    cfg = RigConfig(num_envs=args.num_envs, num_threads=args.num_threads,
                    episode_steps=args.episode_steps, cmd_min=args.cmd_min, cmd_max=args.cmd_max,
                    cmd_period=args.cmd_period, rail_friction=args.rail_friction,
                    dr_enabled=args.dr, seed=args.seed)
    rewards.apply_preset(cfg, args.reward)       # the critic's inputs include the scored speed
    rewards.apply_overrides(cfg, args.set)
    cfg.obs_cmd_scale = cfg.cmd_max               # as train.py
    return cfg


def collect(args) -> None:
    cfg = collect_config(args)
    env = SingleLegRigEnv(cfg)
    env.reset()
    gait = ScriptedGait(env.num_envs)
    rng = np.random.default_rng(args.seed + 1)
    obs, critic, act = [], [], []
    moved = asked = 0.0
    fails = timeouts = 0
    try:
        for _ in range(args.steps):
            a = np.stack([gait.action(env, i) for i in range(env.num_envs)])
            obs.append(env.obs.copy())
            critic.append(env.critic_obs.copy())
            act.append(a)
            noisy = np.clip(a + rng.normal(0, args.noise, a.shape), -1, 1)
            target = env.cmd_raw.copy()
            _, _, done, extras = env.step(torch.from_numpy(noisy.astype(np.float32)))
            vx = np.array([d.qvel[0] for d in env.datas])
            moved += float(np.sum(vx * np.sign(target))) * env.dt
            asked += float(np.sum(np.abs(target))) * env.dt
            done = done.numpy().astype(bool)
            to = extras["time_outs"].numpy().astype(bool)
            fails += int((done & ~to).sum())
            timeouts += int(to.sum())
            for i in np.flatnonzero(done):
                gait.reset(i)
    finally:
        env.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.concatenate(obs).astype(np.float32),
                        critic=np.concatenate(critic).astype(np.float32),
                        act=np.concatenate(act).astype(np.float32))
    out.with_suffix(".json").write_text(json.dumps({"environment": asdict(cfg),
                                                    "noise": args.noise}, indent=2))
    print(f"* {len(obs) * cfg.num_envs} samples -> {out}; scripted walk covered "
          f"{moved / max(asked, 1e-9):.2f} of the asked distance; episodes failed {fails}, "
          f"timed out {timeouts}")


def fit(args) -> None:
    from rsl_rl.runners import OnPolicyRunner
    from single_leg_demo.train import training_config

    data = [np.load(p) for p in args.data]
    X = torch.from_numpy(np.concatenate([d["obs"] for d in data]))
    C = torch.from_numpy(np.concatenate([d["critic"] for d in data]))
    Y = torch.from_numpy(np.concatenate([d["act"] for d in data]))
    saved = json.loads(Path(args.data[0]).with_suffix(".json").read_text())["environment"]
    cfg = RigConfig(**{**saved, "num_envs": 2, "num_threads": 1})
    env = SingleLegRigEnv(cfg)
    if env.num_obs != X.shape[1] or env.num_critic_obs != C.shape[1]:
        raise SystemExit(f"data has {X.shape[1]}/{C.shape[1]} inputs, env {env.num_obs}/{env.num_critic_obs}")
    torch.manual_seed(args.seed)
    runner = OnPolicyRunner(env, training_config("clone", 24), None, device="cpu")
    actor, critic = runner.alg.actor, runner.alg.critic
    for net, data_x in ((actor, X), (critic, C)):
        nz = net.obs_normalizer
        mean, var = data_x.mean(0, keepdim=True), data_x.var(0, keepdim=True)
        nz._mean.copy_(mean)
        nz._var.copy_(var)
        nz._std.copy_(var.sqrt())
        nz.count.fill_(len(data_x))
    opt = torch.optim.Adam(actor.mlp.parameters(), lr=args.lr)
    n = len(X)
    g = torch.Generator().manual_seed(args.seed)
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for b in range(0, n, args.batch):
            idx = perm[b:b + args.batch]
            pred = actor(TensorDict({"policy": X[idx]}, batch_size=[len(idx)]))  # same path as inference
            loss = ((pred - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep in (0, args.epochs // 4, args.epochs // 2, args.epochs - 1):
            print(f"epoch {ep + 1}: action MSE {tot / n:.4f} (label variance {Y.var(0).mean():.4f})")
    dist = actor.distribution
    std_param = getattr(dist, "std_param", None)
    (std_param if std_param is not None else dist.std).data.fill_(args.std)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner.current_learning_iteration = 0
    runner.save(str(out / "model_0.pt"))
    (out / "config.json").write_text(json.dumps(
        {"environment": asdict(cfg),
         "note": f"behaviour clone of the scripted side-swing gait from {', '.join(args.data)}"},
        indent=2))
    env.close()
    print(f"* saved {out / 'model_0.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="run the scripted gait and record (observation, action)")
    c.add_argument("--out", required=True)
    c.add_argument("--steps", type=int, default=2400, help="control steps (x num-envs samples)")
    c.add_argument("--num-envs", type=int, default=64)
    c.add_argument("--num-threads", type=int, default=8)
    c.add_argument("--episode-steps", type=int, default=600)
    c.add_argument("--cmd-min", type=float, default=0.05)
    c.add_argument("--cmd-max", type=float, default=0.08)
    c.add_argument("--cmd-period", type=float, default=4.0)
    c.add_argument("--rail-friction", type=float, default=-1.0, help="N; < 0 keeps rig.xml")
    c.add_argument("--dr", action="store_true", help="domain randomization at full range")
    c.add_argument("--noise", type=float, default=0.15, help="action noise on the executed action")
    c.add_argument("--reward", default=rewards.DEFAULT_PRESET, choices=sorted(rewards.PRESETS))
    c.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    c.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fit", help="fit the actor to collected data and save a checkpoint")
    f.add_argument("data", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--batch", type=int, default=1024)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--std", type=float, default=0.3, help="action noise std the fine-tune starts from")
    f.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    collect(args) if args.cmd == "collect" else fit(args)


if __name__ == "__main__":
    main()
