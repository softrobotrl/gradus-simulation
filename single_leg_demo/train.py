"""PPO training for the demo leg on the one-dimensional sliding rig.

Example::

    python -m single_leg_demo.train --num-envs 64 --iterations 1500

The command is a velocity that flips sign every ``cmd_period`` seconds, so the
policy learns to pump the rail back and forth instead of walking in one
direction.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from single_leg_demo.rig_env import RigConfig, SingleLegRigEnv


def training_config(name: str, steps: int) -> dict:
    return {
        "algorithm": {
            "class_name": "genesis_leg.runner:LegPPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.5,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": True,
        },
        # actor: hardware-measurable state; critic: full simulator state (training only)
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": steps,
        "save_interval": 50,
        "run_name": name,
        "logger": "tensorboard",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--steps-per-env", type=int, default=24)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--episode-steps", type=int, default=400)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--resume", type=Path, default=None,
                   help="continue from a checkpoint; --iterations is additional")
    p.add_argument("--warmup", type=int, default=50,
                   help="iterations at a reduced command before the full range")

    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--std", type=float, default=None,
                   help="freeze the policy action std at this value")
    p.add_argument("--residual-scale", type=float, default=None)
    p.add_argument("--cmd-min", type=float, default=None)
    p.add_argument("--cmd-max", type=float, default=None)
    p.add_argument("--cmd-period", type=float, default=None)
    p.add_argument("--vel-weight", type=float, default=None)
    p.add_argument("--vel-sigma", type=float, default=None)
    p.add_argument("--vel-l2-weight", type=float, default=None)
    p.add_argument("--progress-weight", type=float, default=None)
    p.add_argument("--action-l2-weight", type=float, default=None)
    p.add_argument("--reset-noise", type=float, default=None)
    p.add_argument("--bend-limit", type=float, default=None)
    p.add_argument("--obs-mode", choices=("hardware", "full"), default=None,
                   help="actor observation: hardware sensors (default) or full sim state")
    p.add_argument("--obs-history", type=int, default=None,
                   help="hardware steps stacked for the actor (default 8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("logs/single_leg_demo") / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = RigConfig(
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        episode_steps=args.episode_steps,
        seed=args.seed,
    )
    for key in ("residual_scale", "cmd_min", "cmd_max", "cmd_period",
                "vel_weight", "vel_sigma", "vel_l2_weight", "progress_weight",
                "action_l2_weight", "reset_noise", "bend_limit",
                "obs_mode", "obs_history"):
        value = getattr(args, key)
        if value is not None:
            cfg.__dict__[key] = value
    # Fixed command normaliser for the whole run; keeping it independent of
    # the per-segment sampled magnitude makes the observation well defined.
    cfg.obs_cmd_scale = cfg.cmd_max

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    env = SingleLegRigEnv(cfg)

    train_cfg = training_config(name, args.steps_per_env)
    settings = {
        "environment": asdict(cfg),
        "training": train_cfg,
        "device": args.device,
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))
    print(f"* run: {run_dir}  envs={env.num_envs} obs={env.num_obs} "
          f"({cfg.obs_mode}) critic_obs={env.num_critic_obs} "
          f"threads={env.num_threads} device={args.device}")

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=args.device)
    if args.resume:
        runner.load(str(args.resume), map_location=args.device)
        print(f"* resumed {args.resume} at iteration {runner.current_learning_iteration}")
    if args.entropy_coef is not None:
        runner.alg.entropy_coef = args.entropy_coef
    if args.learning_rate is not None:
        runner.alg.learning_rate = args.learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = args.learning_rate
    if args.std is not None:
        dist = runner.alg.actor.distribution
        std_param = getattr(dist, "std_param", None)
        if std_param is not None:
            std_param.data.fill_(args.std)
            std_param.requires_grad_(False)
        else:
            dist.std.data.fill_(args.std)
            dist.std.requires_grad_(False)
        print(f"* fixed action std at {args.std}")

    full_min, full_max = cfg.cmd_min, cfg.cmd_max
    try:
        if args.resume or args.warmup <= 0:
            runner.learn(args.iterations, init_at_random_ep_len=True)
        else:
            env.cfg["cmd_min"] = min(0.05, full_min)
            env.cfg["cmd_max"] = min(0.08, full_max)
            first = min(args.warmup, args.iterations)
            print(f"* warmup: {first} iterations at cmd={env.cfg['cmd_min']:.2f}.."
                  f"{env.cfg['cmd_max']:.2f} m/s")
            runner.learn(first, init_at_random_ep_len=True)
            remaining = args.iterations - first
            if remaining > 0:
                env.cfg["cmd_min"] = full_min
                env.cfg["cmd_max"] = full_max
                print(f"* full: {remaining} iterations at cmd={full_min:.2f}..{full_max:.2f} m/s")
                runner.learn(remaining)
    finally:
        final = run_dir / f"model_{runner.current_learning_iteration}.pt"
        try:
            runner.save(str(final))
            print(f"* saved {final}")
        except Exception as exc:  # keep the last periodic checkpoint usable
            print(f"* final save failed: {exc}")
        env.close()


if __name__ == "__main__":
    main()
