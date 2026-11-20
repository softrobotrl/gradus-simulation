"""Robustness sweep: how a trained policy copes as one physics parameter changes.

For each setting, one parameter is fixed through ``dr_override`` and the rest
stay nominal; ``--episodes`` episodes run in parallel with the same commands
and seed for every setting, so rows (and checkpoints) are comparable.  Some
values go past the training ranges on purpose, to show where the policy breaks.

Example::

    python -m single_leg_demo.robustness \
        --ckpt logs/single_leg_demo/hw_dr/model_999.pt --episodes 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from single_leg_demo.evaluate import load_policy, saved_env_settings
from single_leg_demo.rig_env import RigConfig, SingleLegRigEnv

# Everything nominal, so that with dr_enabled only sensor noise and pushes remain.
NOMINAL = {
    "friction": 1.0, "rail_friction": 1.0, "rail_damping": 1.0, "rail_height": 0.0,
    "carriage_mass": 1.0,
    "segment_mass": 1.0, "stiffness": 1.0, "joint_damping": 1.0, "servo_kp": 1.0,
    "servo_strength": 1.0, "servo_calib": 0.0, "battery": 1.0, "sag": 0.0, "delay": 0,
    "load_scale": 1.0,
}

# (label, value shown, dr_override, dr_enabled)
SWEEP = (
    [("nominal", "-", {}, False)]
    + [("friction", v, {"friction": v}, False) for v in (0.4, 0.7, 1.0, 1.3)]
    + [("rail_friction", v, {"rail_friction": v}, False) for v in (0.5, 2.0, 3.0)]
    + [("rail_height_mm", v, {"rail_height": v / 1000}, False) for v in (-10, -5, 0.5, 1.5)]
    + [("carriage_mass", v, {"carriage_mass": v}, False) for v in (0.8, 1.2)]
    + [("battery", v, {"battery": v}, False) for v in (0.7, 0.85, 1.0)]
    + [("sag", v, {"sag": v}, False) for v in (0.25, 0.5)]
    + [("delay_ms", 2 * v, {"delay": v}, False) for v in (5, 10, 15)]
    + [("servo_kp", v, {"servo_kp": v}, False) for v in (0.6, 1.4)]
    + [("noise+pushes", "on", dict(NOMINAL), True)]
    + [("all randomized", "train ranges", {}, True)]
)


def run_setting(ckpt: Path, settings: dict, override: dict, dr: bool, args, policy):
    cfg = RigConfig(
        num_envs=args.episodes, num_threads=min(args.episodes, args.num_threads),
        episode_steps=args.steps, seed=args.seed,
        cmd_min=args.cmd_min, cmd_max=args.cmd_max,
        dr_enabled=dr, dr_override=override, **settings,
    )
    env = SingleLegRigEnv(cfg)
    try:
        if policy is None:
            policy = load_policy(env, ckpt)
        obs = env.get_observations()
        alive = np.ones(env.num_envs, dtype=bool)
        failed = np.zeros(env.num_envs, dtype=bool)
        survived = np.zeros(env.num_envs)
        err_sum = np.zeros(env.num_envs)
        for _ in range(args.steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, done, extras = env.step(actions, auto_reset=False)
            vx = np.array([d.qvel[0] for d in env.datas])
            err_sum += np.where(alive, np.abs(vx - env.cmd), 0.0)
            survived += alive
            done = done.numpy().astype(bool)
            failed |= alive & done & ~extras["time_outs"].numpy().astype(bool)
            alive &= ~done
            if not alive.any():
                break
        return {
            "track_err": float(np.mean(err_sum / np.maximum(survived, 1))),
            "fail_rate": float(failed.mean()),
            "survival_s": float(survived.mean() * env.dt),
        }, policy
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=400, help="steps per episode (20 ms each)")
    p.add_argument("--cmd-min", type=float, default=None, help="default: the run's training range")
    p.add_argument("--cmd-max", type=float, default=None)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=Path, default=None, help="CSV path")
    args = p.parse_args()

    settings = saved_env_settings(args.ckpt)
    saved = {}
    train_cfg = args.ckpt.parent / "config.json"
    if train_cfg.exists():
        saved = json.loads(train_cfg.read_text()).get("environment", {})
    if args.cmd_min is None:
        args.cmd_min = float(saved.get("cmd_min", RigConfig.cmd_min))
    if args.cmd_max is None:
        args.cmd_max = float(saved.get("cmd_max", RigConfig.cmd_max))

    print(f"* {args.ckpt}  episodes={args.episodes} x {args.steps * 0.02:.0f} s  "
          f"cmd={args.cmd_min:.2f}..{args.cmd_max:.2f} m/s  obs_mode={settings['obs_mode']}")
    print(f"{'setting':<16}{'value':>14}{'track err m/s':>16}{'fail rate':>12}{'survival s':>12}")
    rows, policy = [], None
    for label, value, override, dr in SWEEP:
        res, policy = run_setting(args.ckpt, settings, override, dr, args, policy)
        rows.append({"setting": label, "value": value, **res})
        print(f"{label:<16}{str(value):>14}{res['track_err']:>16.3f}"
              f"{res['fail_rate']:>12.2f}{res['survival_s']:>12.1f}")

    out = args.out or args.ckpt.parent / f"robustness_{args.ckpt.stem}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"* wrote {out}")


if __name__ == "__main__":
    main()
