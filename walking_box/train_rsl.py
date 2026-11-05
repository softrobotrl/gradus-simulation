"""rsl-rl PPO for the CPU quadruped env.

Example::

    python -m walking_box.train_rsl --num-envs 64 --iterations 2000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from walking_box.cpu_env import EnvConfig, QuadrupedCpuEnv


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
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": steps,
        "save_interval": 50,
        "run_name": name,
        "logger": "tensorboard",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--steps-per-env", type=int, default=24)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--episode-steps", type=int, default=400)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("logs/walking_box") / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        episode_steps=args.episode_steps,
    )
    env = QuadrupedCpuEnv(cfg)
    train_cfg = training_config(name, args.steps_per_env)
    settings = {
        "environment": asdict(cfg),
        "training": train_cfg,
        "device": args.device,
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))
    print(f"* run: {run_dir}  envs={env.num_envs} obs={env.num_obs} "
          f"threads={env.num_threads} device={args.device}")

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=args.device)
    try:
        # curriculum: stand -> slow walk -> full command range
        env.cfg["cmd_max"] = 0.0
        env.cfg["lift_weight"] = 0.0
        first = min(300, args.iterations)
        runner.learn(first, init_at_random_ep_len=True)
        if args.iterations > first:
            env.cfg["cmd_max"] = 0.10
            env.cfg["lift_weight"] = 0.2
            print("* curriculum: cmd_max=0.10, foot lift on")
            second = min(400, args.iterations - first)
            runner.learn(second)
            remaining = args.iterations - first - second
            if remaining > 0:
                env.cfg["cmd_max"] = 0.25
                print("* curriculum: cmd_max=0.25")
                runner.learn(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()
