"""Run a walking-box checkpoint from perturbed initial states.

Example::

    python -m walking_box.robustness \
        --ckpt logs/walking_box/walk_diag_yaw/model_1600.pt \
        --cmd 0.25 --gait-amp 0.8 --period 0.86 --seeds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from walking_box.cpu_env import EnvConfig, QuadrupedCpuEnv
from walking_box.train_rsl import training_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path,
                   default=Path("logs/walking_box/walk_diag_yaw/model_1600.pt"))
    p.add_argument("--cmd", type=float, default=0.25)
    p.add_argument("--gait-amp", type=float, default=0.8)
    p.add_argument("--period", type=float, default=0.86)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--noise", type=float, default=1.0,
                   help="scale the env reset perturbation")
    args = p.parse_args()

    cfg = EnvConfig(
        num_envs=1,
        num_threads=1,
        episode_steps=args.steps,
        gait_amp=args.gait_amp,
        gait_mode="diagonal",
        action_dim=32,
        period=args.period,
        reset_noise=args.noise,
    )
    env = QuadrupedCpuEnv(cfg)
    runner = OnPolicyRunner(env, training_config("eval", 24), None, device="cpu")
    runner.load(str(args.ckpt), map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    rows = []
    for seed in range(args.seeds):
        env.rng = np.random.default_rng(seed)
        env.reset()
        env.cmd[0, 0] = args.cmd
        env.cmd[0, 1:] = 0.0
        env.obs[0] = env._obs_one(0)
        obs = env.get_observations()

        failed = False
        ups, zs = [], []
        for _ in range(args.steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, done, extras = env.step(actions, auto_reset=False)
            ups.append(float(extras["log"]["/up_mean"]))
            zs.append(float(extras["log"]["/height_mean"]))
            if bool(done[0]):
                failed = not bool(extras["time_outs"][0])
                break
        d = env.datas[0]
        dx = float(d.qpos[0] - env.qpos0[0])
        dy = float(d.qpos[1] - env.qpos0[1])
        rows.append((seed, len(ups), dx, dy, min(ups), min(zs), failed))

    env.close()

    print("seed steps     dx     dy   up_min  z_min  failed")
    for r in rows:
        print(f"{r[0]:4d} {r[1]:5d} {r[2]:6.3f} {r[3]:6.3f} "
              f"{r[4]:7.3f} {r[5]:6.3f}  {r[6]}")
    good = [r for r in rows if not r[6]]
    print(f"survived {len(good)}/{len(rows)}", end="")
    if good:
        print(f"  mean_dx={np.mean([r[2] for r in good]):.3f} m"
              f"  mean_steps={np.mean([r[1] for r in good]):.0f}")
    else:
        print()


if __name__ == "__main__":
    main()
