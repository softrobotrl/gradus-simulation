"""Run a saved policy or a baseline controller, with an optional viewer."""

import argparse
import json
import math
from pathlib import Path

import torch


def scripted_action(env, time, amplitude=0.6, hz=0.5):
    """Sweep the tip in a circle, lifting the board on the forward stroke."""
    holes = env.offsets[:, 2, :2]
    unit = holes / torch.linalg.vector_norm(holes, dim=-1, keepdim=True)
    phase = 2 * math.pi * hz * time
    bend = env._tensor([amplitude * math.sin(phase), amplitude * (1 - math.cos(phase))])
    angles = unit @ bend
    low, high = env.robot_cfg.servo.angle_range
    return torch.where(angles >= 0, angles / high, -angles / low).expand(env.num_envs, -1)


def evaluate(env, policy):
    """Measure the first episode from each parallel environment."""
    obs = env.reset()
    initial = env.robot.get_qpos()[:, env.board_qs[:2]].clone()
    final = initial.clone()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device)
    returns = torch.zeros_like(lengths)
    failed = torch.zeros_like(alive)
    max_tension = 0.0
    with torch.no_grad():
        for step in range(env.max_episode_length):
            obs, reward, done, extras = env.step(policy(obs, step * env.dt))
            returns[alive] += reward[alive]
            lengths[alive] += 1
            final[alive] = extras["final_observation"][alive][:, env.board_qs[:2]]
            failed |= alive & done & ~extras["time_outs"].bool()
            # Finished environments auto-reset; ignore their subsequent episodes.
            max_tension = max(max_tension, float(extras["peak_tension"][alive].max()))
            alive &= ~done
            if not alive.any():
                break
    seconds = lengths * env.dt
    return {
        "episodes": env.num_envs,
        "return_mean": returns.mean().item(),
        "episode_seconds_mean": seconds.mean().item(),
        "board_speed_mm_s": ((final[:, 0] - initial[:, 0]) / seconds).mean().item() * 1000,
        "lateral_travel_mm": (final[:, 1] - initial[:, 1]).abs().mean().item() * 1000,
        "failure_rate": failed.float().mean().item(),
        "peak_tension_N": max_tension,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--baseline", choices=["zero", "random", "scripted"], default="zero")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--gait-amplitude", type=float, default=0.6, help="Scripted drum amplitude in radians"
    )
    parser.add_argument("--gait-hz", type=float, default=0.5)
    parser.add_argument(
        "--min-speed-mm-s",
        type=float,
        help="Require this mean forward speed and no physical failures",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.gait_amplitude <= 0 or args.gait_hz <= 0:
        parser.error("episodes, gait-amplitude, and gait-hz must be positive")

    settings = None
    if args.checkpoint:
        if not args.checkpoint.is_file():
            parser.error(
                f"Checkpoint not found: {args.checkpoint}. "
                "Use a checkpoint from an existing run in logs/genesis_leg/."
            )
        config_path = args.checkpoint.parent / "config.json"
        if not config_path.is_file():
            parser.error(
                f"Run configuration not found: {config_path}. "
                "Copy config.json alongside the checkpoint when transferring a run."
            )
        settings = json.loads(config_path.read_text())

    import genesis as gs

    from genesis_leg.env import GenesisLegEnv, LegConfig
    from genesis_leg.train import initialize

    cfg = LegConfig(**settings["environment"]) if settings is not None else LegConfig()
    cfg.num_envs = 1 if args.view else args.episodes
    initialize(args.backend, 12345)
    env = GenesisLegEnv(cfg, show_viewer=args.view)
    if args.checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, settings["training"], log_dir=None, device=gs.device)
        runner.load(str(args.checkpoint), map_location=str(gs.device))
        actor = runner.get_inference_policy(device=gs.device)

        def policy(obs, time):
            return actor(obs)

    elif args.baseline == "scripted":

        def policy(obs, time):
            return scripted_action(env, time, args.gait_amplitude, args.gait_hz)

    elif args.baseline == "random":

        def policy(obs, time):
            return torch.rand_like(env.targets) * 2 - 1

    else:

        def policy(obs, time):
            return torch.zeros_like(env.targets)

    results = evaluate(env, policy)
    print(json.dumps(results, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    if args.min_speed_mm_s is not None:
        if results["board_speed_mm_s"] < args.min_speed_mm_s or results["failure_rate"] > 0:
            parser.exit(1, "FAIL: forward-speed/failure-rate requirement not met\n")
        print("PASS: forward-speed and failure-rate requirement")


if __name__ == "__main__":
    main()
