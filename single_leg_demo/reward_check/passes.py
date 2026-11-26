"""Pass-by-pass gait report: how a checkpoint actually walks, no reward involved.

    python -m single_leg_demo.reward_check.passes logs/single_leg_demo/RUN/model_300.pt ...
    python -m single_leg_demo.reward_check.passes --group logs/single_leg_demo/RUN_s*/model_300.pt

The command reverses every --period seconds; each stretch between reversals
is a pass.  For every pass it measures distance moved vs asked, and the share
of time the carriage moves the commanded way, stands still (|speed| < 1 cm/s)
or moves the wrong way, plus foot lifts above 1 cm.  The leg starts once in
each direction.

Why this exists: one averaged number (distance / asked over a whole run)
rated a policy that pushes hard and then idles as good.  Judge checkpoints
by these columns, then watch the finalists.

--group averages all given checkpoints into one row (e.g. the seeds of one
reward) instead of printing one row each.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import torch

from single_leg_demo.evaluate import load_policy, saved_env_settings
from single_leg_demo.rig_env import FOOT_RADIUS, RigConfig, SingleLegRigEnv

STILL = 0.01     # m/s


def run(ckpt: Path, sign: float, cmd: float, period: float, steps: int, rail: float,
        dr_seed: int | None = None) -> dict:
    settings = saved_env_settings(ckpt)
    settings.update(rail_friction=rail, cmd_ramp=0.3)
    env = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=steps, cmd_min=cmd,
                                    cmd_max=cmd, cmd_period=period, cmd_period_jitter=0.0,
                                    cmd_resample=False, reset_noise=0.0,
                                    seed=5 if dr_seed is None else dr_seed,
                                    dr_enabled=dr_seed is not None, **settings))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            policy = load_policy(env, ckpt)
        env.reset()
        env.cmd_sign0[0] = sign
        env.cmd_mag[0] = cmd
        env.cmd_period[0] = period
        env.cmd[0] = 0.0
        env._set_cmd(0)
        env._push_obs(0, fresh=True)
        obs = env.get_observations()
        xs, vs, cs, fz = [], [], [], []
        failed = False
        for _ in range(steps):
            with torch.no_grad():
                act = policy(obs)
            obs, _, done, extras = env.step(act, auto_reset=False)
            d = env.datas[0]
            xs.append(float(d.qpos[0]))
            vs.append(float(d.qvel[0]))
            cs.append(float(env.cmd_raw[0]))
            fz.append(float(d.site_xpos[env.tip_id][2]) - FOOT_RADIUS)
            if bool(done[0]):
                failed = not bool(extras["time_outs"][0])
                break
    finally:
        env.close()
    xs, vs, cs, fz = map(np.array, (xs, vs, cs, fz))
    cut = np.r_[0, np.flatnonzero(np.diff(np.sign(cs))) + 1, len(xs)]
    passes = []
    for a, b in zip(cut[:-1], cut[1:]):
        if b - a < 10:
            continue
        sgn = np.sign(cs[a])
        v = vs[a:b] * sgn
        passes.append(dict(
            dist=(xs[b - 1] - xs[a]) * sgn / max(abs(cs[a]) * (b - a) * env.dt, 1e-9),
            right=float(np.mean(v > STILL)), still=float(np.mean(np.abs(v) <= STILL)),
            wrong=float(np.mean(v < -STILL)),
            lifts=int(np.sum(np.diff((fz[a:b] > 0.01).astype(int)) == 1))))
    return dict(passes=passes, failed=failed, seconds=len(xs) * env.dt)


def summarise(results: list[dict]) -> dict:
    p = [q for r in results for q in r["passes"]]
    mean = {k: float(np.mean([q[k] for q in p])) for k in ("dist", "right", "still", "wrong", "lifts")}
    mean["dist_spread"] = float(np.std([q["dist"] for q in p]))
    mean["failed"] = sum(r["failed"] for r in results)
    mean["runs"] = len(results)
    mean["seconds"] = float(np.mean([r["seconds"] for r in results]))
    return mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", type=Path)
    p.add_argument("--group", action="store_true", help="one averaged row for all checkpoints")
    p.add_argument("--label", default=None, help="row name with --group")
    p.add_argument("--cmd", type=float, default=0.07)
    p.add_argument("--period", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--rail-friction", type=float, default=3.0)
    p.add_argument("--dr", type=int, default=0, metavar="N",
                   help="also run N randomized-physics episodes per direction (seeds 0..N-1); "
                        "0 = nominal physics only")
    args = p.parse_args()

    print(f"({'randomized physics, ' + str(args.dr) + ' seeds; ' if args.dr else ''}rail {args.rail_friction} N, cmd {args.cmd} m/s, {args.period} s passes, "
          f"{args.steps * 0.02:.0f} s runs, both directions; means over passes)")
    print(f"{'checkpoint':40s} {'dist/asked':>12s} {'right way':>9s} {'still':>6s} {'wrong':>6s} "
          f"{'lifts/pass':>10s} {'failed':>8s} {'ran':>6s}")
    groups = [args.ckpts] if args.group else [[c] for c in args.ckpts]
    for group in groups:
        seeds = list(range(args.dr)) if args.dr else [None]
        res = [run(c, s, args.cmd, args.period, args.steps, args.rail_friction, k)
               for c in group for s in (1.0, -1.0) for k in seeds]
        m = summarise(res)
        name = args.label or (f"{len(group)} checkpoints" if args.group
                              else f"{group[0].parent.name}/{group[0].name}")
        print(f"{name:40s} {m['dist']:6.2f}±{m['dist_spread']:4.2f} {m['right']:9.0%} "
              f"{m['still']:6.0%} {m['wrong']:6.0%} {m['lifts']:10.1f} "
              f"{m['failed']:3d} of {m['runs']:<2d} {m['seconds']:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
