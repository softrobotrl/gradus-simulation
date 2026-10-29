"""Train a PPO policy for the single-leg board-pushing task."""

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path


def training_config(name, steps=24):
    from genesis_samples.go2_train import get_train_cfg

    cfg = get_train_cfg(name)
    cfg["algorithm"]["class_name"] = "genesis_leg.runner:LegPPO"
    cfg["actor"]["hidden_dims"] = [128, 128]
    cfg["critic"]["hidden_dims"] = [128, 128]
    cfg["actor"]["obs_normalization"] = True
    cfg["critic"]["obs_normalization"] = True
    cfg["num_steps_per_env"] = steps
    cfg["save_interval"] = 50
    return cfg


def initialize(backend, seed):
    import genesis as gs
    import torch

    if backend == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot find an NVIDIA GPU. Use --backend cpu for local checks."
        )
    gs.init(
        backend=gs.gpu if backend == "gpu" else gs.cpu,
        seed=seed,
        precision="32",
        logging_level="warning",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Additional iterations when resuming"
    )
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, help="Iterations between checkpoints (default: 50)")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config")
    parser.add_argument("--cable-stiffness", type=float)
    parser.add_argument("--cable-damping", type=float)
    parser.add_argument("--action-filter", type=float)
    parser.add_argument("--episode-s", type=float)
    args = parser.parse_args()
    if (
        args.iterations < 1
        or args.steps_per_env < 4
        or (args.num_envs is not None and args.num_envs < 1)
        or (args.save_interval is not None and args.save_interval < 1)
    ):
        parser.error("Require iterations >= 1, steps-per-env >= 4, num-envs >= 1, save-interval >= 1")
    env_overrides = {
        name: getattr(args, name)
        for name in ("cable_stiffness", "cable_damping", "action_filter", "episode_s")
        if getattr(args, name) is not None
    }
    if args.resume and (args.config or args.run_name or env_overrides):
        parser.error("Resume uses the original run's config and directory")
    if args.resume:
        if not args.resume.is_file():
            parser.error(f"Checkpoint not found: {args.resume}")
        if not (args.resume.parent / "config.json").is_file():
            parser.error(f"Missing config.json alongside {args.resume}")

    import genesis as gs

    from genesis_leg.env import GenesisLegEnv, LegConfig
    from genesis_leg.runner import LegRunner

    if args.resume:
        run_dir = args.resume.resolve().parent
        settings = json.loads((run_dir / "config.json").read_text())
        env_cfg = LegConfig(**settings["environment"])
        seed = settings["seed"] if args.seed is None else args.seed
        train_cfg = settings["training"]
        if train_cfg["algorithm"]["class_name"] == "PPO":
            train_cfg["algorithm"]["class_name"] = "genesis_leg.runner:LegPPO"
    else:
        name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        if Path(name).name != name or name in (".", ".."):
            parser.error("run-name must be a directory name")
        run_dir = Path("logs/genesis_leg") / name
        if run_dir.exists():
            parser.error(f"{run_dir} exists; choose another run-name or --resume a checkpoint")
        env_cfg = LegConfig()
        seed = 0 if args.seed is None else args.seed
        if args.config:
            env_cfg.robot_config = args.config
        train_cfg = training_config(name, args.steps_per_env)
    if args.num_envs is not None:
        env_overrides["num_envs"] = args.num_envs
    env_cfg = replace(env_cfg, **env_overrides)
    if args.save_interval is not None:
        train_cfg["save_interval"] = args.save_interval

    initialize(args.backend, seed)
    env = GenesisLegEnv(env_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "environment": asdict(env_cfg),
        "training": train_cfg,
        "seed": seed,
        "versions": {
            package: version(package)
            for package in ("genesis-world", "rsl-rl-lib", "torch", "tensordict")
        },
    }
    if not args.resume:
        (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    else:
        settings["checkpoint"] = str(args.resume.resolve())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (run_dir / f"resume_{stamp}.json").write_text(json.dumps(settings, indent=2) + "\n")
    runner = LegRunner(env, train_cfg, str(run_dir), device=gs.device)
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        runner.load(str(args.resume), map_location=str(gs.device))
        runner.current_learning_iteration += 1
    print(
        f"Training in {run_dir}: {env.num_envs} environments, "
        f"{env.num_obs} observations, {env.num_actions} actions"
    )
    try:
        try:
            runner.learn(args.iterations, init_at_random_ep_len=True)
        except KeyboardInterrupt:
            print("Interrupted; saving latest.pt")
        runner.save(str(run_dir / "latest.pt"))
    finally:
        if runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()
    print(f"Saved {run_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
