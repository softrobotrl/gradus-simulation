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

import numpy as np
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
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--steps-per-env", type=int, default=24)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--episode-steps", type=int, default=400)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--resume", type=Path, default=None,
                   help="continue from a checkpoint; --iterations is additional")
    p.add_argument("--entropy-coef", type=float, default=None,
                   help="override PPO entropy coefficient after loading")
    p.add_argument("--learning-rate", type=float, default=None,
                   help="override PPO learning rate after loading")
    p.add_argument("--std", type=float, default=None,
                   help="freeze the policy action std at this value")
    p.add_argument("--zero-actor-output", action="store_true",
                   help="zero the actor's last layer so training starts from the nominal gait")
    p.add_argument("--fall-penalty", type=float, default=None,
                   help="override the reward fall penalty")
    p.add_argument("--tilt-weight", type=float, default=None,
                   help="override the reward tilt weight")
    p.add_argument("--reset-noise", type=float, default=None,
                   help="scale initial-state perturbations (0 disables)")
    p.add_argument("--gait-amp", type=float, default=None,
                   help="override the nominal gait amplitude")
    p.add_argument("--period", type=float, default=None,
                   help="override the nominal gait period")
    p.add_argument("--swing-frac", type=float, default=None,
                   help="override the swing fraction of the gait cycle")
    p.add_argument("--stance-amp", type=float, default=None,
                   help="override the stance-phase back-tendon pull")
    p.add_argument("--residual-scale", type=float, default=None,
                   help="override the policy residual scale")
    p.add_argument("--cmd-min", type=float, default=None,
                   help="override the minimum forward command")
    p.add_argument("--cmd-max", type=float, default=None,
                   help="override the maximum forward command")
    p.add_argument("--vel-weight", type=float, default=None,
                   help="override the velocity-tracking reward weight")
    p.add_argument("--vel-sigma", type=float, default=None,
                   help="override the velocity-tracking tolerance")
    p.add_argument("--progress-weight", type=float, default=None,
                   help="override the forward-progress reward weight")
    p.add_argument("--lat-weight", type=float, default=None,
                   help="override the lateral-velocity penalty weight")
    p.add_argument("--yaw-weight", type=float, default=None,
                   help="override the yaw-rate penalty weight")
    p.add_argument("--heading-weight", type=float, default=None,
                   help="override the heading penalty weight")
    p.add_argument("--slip-weight", type=float, default=None,
                   help="override the stance-slip penalty weight")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("logs/walking_box") / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        episode_steps=args.episode_steps,
        gait_mode="diagonal",
    )
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    env = QuadrupedCpuEnv(cfg)
    if args.reset_noise is not None:
        env.cfg["reset_noise"] = args.reset_noise
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
    if args.zero_actor_output:
        last = runner.alg.actor.mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        print("* zeroed actor output layer (start from the nominal gait)")
    try:
        if args.resume:
            # Fine-tune directly in the full walking regime; re-running the
            # stand phase would pull the policy back to standing.
            env.cfg["gait_mode"] = "diagonal"
            env.cfg["cmd_min"] = 0.10
            env.cfg["cmd_max"] = 0.25
            env.cfg["vel_weight"] = 2.0
            env.cfg["progress_weight"] = 0.7
            env.cfg["survival_bonus"] = 0.0
            env.cfg["lift_weight"] = 0.3
            env.cfg["gait_amp"] = 0.7
            env.cfg["period"] = 0.8
            env.cfg["lift_height"] = 0.035
            env.cfg["slip_weight"] = 0.3
            env.cfg["lat_weight"] = 2.0
            env.cfg["yaw_weight"] = 1.5
            env.cfg["heading_weight"] = 1.0
            if args.fall_penalty is not None:
                env.cfg["fall_penalty"] = args.fall_penalty
            if args.tilt_weight is not None:
                env.cfg["tilt_weight"] = args.tilt_weight
            if args.reset_noise is not None:
                env.cfg["reset_noise"] = args.reset_noise
            if args.gait_amp is not None:
                env.cfg["gait_amp"] = args.gait_amp
            if args.period is not None:
                env.cfg["period"] = args.period
            if args.swing_frac is not None:
                env.cfg["swing_frac"] = args.swing_frac
            if args.stance_amp is not None:
                env.cfg["stance_amp"] = args.stance_amp
            if args.residual_scale is not None:
                env.cfg["residual_scale"] = args.residual_scale
            if args.cmd_min is not None:
                env.cfg["cmd_min"] = args.cmd_min
            if args.cmd_max is not None:
                env.cfg["cmd_max"] = args.cmd_max
            if args.vel_weight is not None:
                env.cfg["vel_weight"] = args.vel_weight
            if args.vel_sigma is not None:
                env.cfg["vel_sigma"] = args.vel_sigma
            if args.progress_weight is not None:
                env.cfg["progress_weight"] = args.progress_weight
            if args.lat_weight is not None:
                env.cfg["lat_weight"] = args.lat_weight
            if args.yaw_weight is not None:
                env.cfg["yaw_weight"] = args.yaw_weight
            if args.heading_weight is not None:
                env.cfg["heading_weight"] = args.heading_weight
            if args.slip_weight is not None:
                env.cfg["slip_weight"] = args.slip_weight
            print(f"* fine-tune: diagonal trot, cmd={env.cfg['cmd_min']:.2f}.."
                  f"{env.cfg['cmd_max']:.2f}, amp={env.cfg['gait_amp']:.2f}, "
                  f"period={env.cfg['period']:.2f}, swing={env.cfg['swing_frac']:.2f}, "
                  f"stance={env.cfg['stance_amp']:.2f}, "
                  f"reset_noise={env.cfg['reset_noise']:.2f}, stronger yaw/lateral")
            runner.learn(args.iterations)
            return

        # Curriculum: brief stand, then slow diagonal trot, then full range.
        # The stand phase is deliberately short so the policy does not settle
        # into a stable standing local optimum.
        env.cfg["gait_mode"] = "diagonal"
        env.cfg["cmd_min"] = 0.0
        env.cfg["cmd_max"] = 0.0
        env.cfg["lift_weight"] = 0.0
        env.cfg["gait_amp"] = 0.0
        first = min(50, args.iterations)
        runner.learn(first, init_at_random_ep_len=True)
        if args.iterations > first:
            env.cfg["cmd_min"] = 0.05
            env.cfg["cmd_max"] = 0.15
            env.cfg["vel_weight"] = 1.5
            env.cfg["progress_weight"] = 0.5
            env.cfg["survival_bonus"] = 0.0
            env.cfg["lift_weight"] = 0.3
            env.cfg["gait_amp"] = 0.6
            print("* curriculum: diagonal trot, cmd=0.05..0.15")
            second = min(300, args.iterations - first)
            runner.learn(second)
            remaining = args.iterations - first - second
            if remaining > 0:
                env.cfg["cmd_min"] = 0.10
                env.cfg["cmd_max"] = 0.25
                env.cfg["vel_weight"] = 2.0
                env.cfg["progress_weight"] = 0.7
                env.cfg["lift_weight"] = 0.3
                env.cfg["gait_amp"] = 0.7
                print("* curriculum: diagonal trot, cmd=0.10..0.25")
                runner.learn(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()
