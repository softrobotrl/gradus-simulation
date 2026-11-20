"""Lockstep scoring: score trained checkpoints under two rewards on the same motion.

    python -m single_leg_demo.reward_check.lockstep --a v1 --b v4 \\
        logs/single_leg_demo/RUN1/model_998.pt logs/single_leg_demo/RUN2/model_500.pt ...
    python -m single_leg_demo.reward_check.lockstep --a v4 --b v4 --set-b accel_weight=1.0 CKPT ...

Each checkpoint drives the leg once in each direction.  Two physically
identical envs step together with the same actions; only fields that change
the reward differ (rewards.REWARD_FIELDS), so A and B score exactly the same
motion (the drift column checks that).  Physics and observations come from
each checkpoint's own run (evaluate.saved_env_settings), with this test's
command and rail friction for all of them.

It also reports measures that do not depend on any reward: distance covered /
asked, foot lifts above 1 cm, failures.  With 4+ checkpoints it prints each
reward's rank correlation (Spearman) with tracking (-|distance/asked - 1|)
and with lifts.  A good reward ranks policies by how well they do the task.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import torch

from single_leg_demo import rewards
from single_leg_demo.evaluate import load_policy, saved_env_settings
from single_leg_demo.rig_env import FOOT_RADIUS, RigConfig, SingleLegRigEnv


def make_pair(ckpt: Path, a: str, set_a: list[str], b: str, set_b: list[str],
              cmd: float, period: float, steps: int, rail: float, cmd_ramp: float):
    settings = saved_env_settings(ckpt)
    settings.update(rail_friction=rail, cmd_ramp=cmd_ramp)
    base = dict(num_envs=1, num_threads=1, episode_steps=steps, cmd_min=cmd, cmd_max=cmd,
                cmd_period=period, cmd_period_jitter=0.0, cmd_resample=False,
                reset_noise=0.0, seed=5, **settings)
    ca = rewards.apply_overrides(rewards.apply_preset(RigConfig(**base), a), set_a)
    ca.curriculum_k0 = 1.0
    ca.cmd_ramp = cmd_ramp                     # the command belongs to the test, not the reward (v1 sets 0)
    cb_full = rewards.apply_overrides(rewards.apply_preset(RigConfig(**base), b), set_b)
    cb = RigConfig(**base)
    for key in vars(ca):                       # B = A except for reward-only fields
        setattr(cb, key, getattr(cb_full, key) if key in rewards.REWARD_FIELDS else getattr(ca, key))
    cb.curriculum_k0 = 1.0
    ignored = sorted(k for k in vars(ca) if k not in rewards.REWARD_FIELDS
                     and getattr(cb_full, k) != getattr(ca, k))
    return ca, cb, ignored


def score(ckpt: Path, ca: RigConfig, cb: RigConfig) -> dict:
    runs = []
    for sign in (1.0, -1.0):
        A, B = SingleLegRigEnv(ca), SingleLegRigEnv(cb)
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # rsl_rl prints the networks
                policy = load_policy(A, ckpt)
            for E in (A, B):
                E.reset()
                E.cmd_sign0[0] = sign
                E.cmd_mag[0] = ca.cmd_max
                E.cmd_period[0] = ca.cmd_period
                E.cmd[0] = 0.0
                E._set_cmd(0)
                E._push_obs(0, fresh=True)
            obs = A.get_observations()
            ra = rb = moved = asked = drift = 0.0
            heights, failed = [], False
            for t in range(ca.episode_steps):
                with torch.no_grad():
                    act = policy(obs)
                target = float(A.cmd_raw[0])
                obs, r1, d1, x1 = A.step(act, auto_reset=False)
                _, r2, d2, _ = B.step(act, auto_reset=False)
                ra += float(r1[0])
                rb += float(r2[0])
                drift = max(drift, float(np.abs(A.datas[0].qpos - B.datas[0].qpos).max()))
                moved += float(A.datas[0].qvel[0]) * np.sign(target) * A.dt
                asked += abs(target) * A.dt
                heights.append(float(A.datas[0].site_xpos[A.tip_id][2]) - FOOT_RADIUS)
                if bool(d1[0]):
                    failed = not bool(x1["time_outs"][0])
                    break
            lifts = int(np.sum(np.diff((np.array(heights) > 0.01).astype(int)) == 1))
            runs.append(dict(a=ra / ca.episode_steps, b=rb / ca.episode_steps,
                             distance=moved / max(asked, 1e-9), lifts=lifts,
                             failed=failed, drift=drift))
        finally:
            A.close()
            B.close()
    out = {k: float(np.mean([r[k] for r in runs])) for k in ("a", "b", "distance", "lifts")}
    out["failed"] = sum(r["failed"] for r in runs)
    out["drift"] = max(r["drift"] for r in runs)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", type=Path)
    p.add_argument("--a", default="v1", choices=sorted(rewards.PRESETS))
    p.add_argument("--b", default=rewards.DEFAULT_PRESET, choices=sorted(rewards.PRESETS))
    p.add_argument("--set-a", action="append", default=[], metavar="FIELD=VALUE")
    p.add_argument("--set-b", action="append", default=[], metavar="FIELD=VALUE")
    p.add_argument("--cmd", type=float, default=0.07)
    p.add_argument("--period", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--rail-friction", type=float, default=3.0)
    p.add_argument("--cmd-ramp", type=float, default=0.3, help="m/s^2, same for all checkpoints")
    args = p.parse_args()

    la = args.a + (" " + " ".join(args.set_a) if args.set_a else "")
    lb = args.b + (" " + " ".join(args.set_b) if args.set_b else "")
    print(f"A = {la}   B = {lb}   (rail {args.rail_friction} N, cmd {args.cmd} m/s, "
          f"{args.period} s passes, {args.steps} steps, both directions)")
    print(f"{'checkpoint':48s} {'A/step':>8s} {'B/step':>8s} {'dist/asked':>10s} "
          f"{'lifts':>6s} {'failed':>6s} {'drift':>8s}")
    rows, warned = [], False
    for ckpt in args.ckpts:
        ca, cb, ignored = make_pair(ckpt, args.a, args.set_a, args.b, args.set_b, args.cmd,
                                    args.period, args.steps, args.rail_friction, args.cmd_ramp)
        if ignored and not warned:
            print(f"* B keeps A's {', '.join(ignored)} (not reward-only fields)")
            warned = True
        r = score(ckpt, ca, cb)
        rows.append(r)
        name = f"{ckpt.parent.name}/{ckpt.name}"
        print(f"{name:48s} {r['a']:+8.3f} {r['b']:+8.3f} {r['distance']:10.2f} "
              f"{r['lifts']:6.1f} {r['failed']:6d} {r['drift']:8.1e}", flush=True)
    if len(rows) >= 4:
        from scipy.stats import spearmanr
        track = [-abs(r["distance"] - 1.0) for r in rows]
        lifts = [r["lifts"] for r in rows]
        for label, key in (("A", "a"), ("B", "b")):
            s = [r[key] for r in rows]
            print(f"rank correlation of {label}'s score with tracking {spearmanr(s, track)[0]:+.2f}, "
                  f"with lifts {spearmanr(s, lifts)[0]:+.2f}")


if __name__ == "__main__":
    main()
