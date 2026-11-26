"""Preference test: score fixed behaviours under a reward, no training.

    python -m single_leg_demo.reward_check.preference                    # the default reward (v5)
    python -m single_leg_demo.reward_check.preference --set accel_weight=1.0
    python -m single_leg_demo.reward_check.preference --compare v1 v4

Five behaviours run on the same physics: the scripted walk (reward_check.gait),
the same walk jumping between keyframes (jerky), the walk plus action noise
(noisy, like PPO exploring), the walk for 1 s then frozen for 1 s, repeated
(burst: what PPO fine-tunes drifted into under v4), and standing still.  The costs run at full
strength (curriculum k = 1, as at the end of training).

A reward worth training must score the walk above standing still (else
FAIL: training will find standing).  It should also score the walk above the
noisy, jerky and burst walks (else WARN: the reward does not separate them).
If the scripted walk itself does not cover the distance on these physics
(e.g. too much rail friction), the test cannot judge the reward (INCONCLUSIVE).

Exit code 1 on FAIL, so it can gate a script or CI.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from single_leg_demo import rewards
from single_leg_demo.reward_check.gait import ScriptedGait
from single_leg_demo.rig_env import FOOT_RADIUS, RigConfig, SingleLegRigEnv

BEHAVIOURS = ("scripted gait", "jerky gait", "noisy gait", "burst gait", "stand still")
BURST = 50           # control steps (1 s) of walking, then as many holding still
NOISE = 0.3          # action noise std of the noisy walk


def make_config(preset: str, overrides: list[str], rail: float, cmd: float, period: float,
                steps: int) -> RigConfig:
    # The scripted walk swings the foot sideways, which the confirmed 4-servo
    # leg cannot do, so the test still runs on the 8-servo rig (it judges the
    # reward, not the leg).  TODO: a forward/back-only scripted walk.
    c = RigConfig(servo_layout="independent", servo_torque=0.0,
                  num_envs=1, num_threads=1, episode_steps=steps, cmd_min=cmd, cmd_max=cmd,
                  cmd_period=period, cmd_period_jitter=0.0, cmd_resample=False,
                  reset_noise=0.0, rail_friction=rail, obs_cmd_scale=max(0.08, cmd))
    rewards.apply_preset(c, preset)
    c.curriculum_k0 = 1.0            # costs at full strength
    rewards.apply_overrides(c, overrides)
    return c


def run_behaviour(cfg: RigConfig, behaviour: str, seed: int = 0) -> dict:
    """Mean reward per step, per-term means and task measures for one behaviour."""
    env = SingleLegRigEnv(cfg)
    try:
        env.reset()
        rng = np.random.default_rng(seed)
        gait = ScriptedGait(1)
        gait.jerky = behaviour == "jerky gait"
        steps = cfg.episode_steps
        total, terms = 0.0, {}
        moved = asked = 0.0
        heights, fails = [], 0
        for t in range(steps):
            if behaviour == "stand still":
                a = np.zeros(env.num_actions)
            elif behaviour == "burst gait" and (t // BURST) % 2 == 1:
                pass                                # hold the last action: the leg freezes
            else:
                a = gait.action(env, 0)
                if behaviour == "noisy gait":
                    a = np.clip(a + rng.normal(0, NOISE, env.num_actions), -1, 1)
            target = float(env.cmd_raw[0])
            _, r, done, extras = env.step(torch.from_numpy(a[None].astype(np.float32)))
            total += float(r[0])
            for k, v in extras["log"].items():
                if k.startswith("/r_"):
                    terms[k[3:]] = terms.get(k[3:], 0.0) + v
            d = env.datas[0]
            moved += float(d.qvel[0]) * np.sign(target) * env.dt
            asked += abs(target) * env.dt
            heights.append(float(d.site_xpos[env.tip_id][2]) - FOOT_RADIUS)
            if bool(done[0]) and t < steps - 1:
                fails += int(not bool(extras["time_outs"][0]))
                gait.reset(0)
        lifts = int(np.sum(np.diff((np.array(heights) > 0.01).astype(int)) == 1))
        return {"reward": total / steps, "terms": {k: v / steps for k, v in terms.items()},
                "distance": moved / max(asked, 1e-9), "lifts": lifts, "failed": fails}
    finally:
        env.close()


def score(cfg: RigConfig, workers: int = 4) -> dict:
    with ProcessPoolExecutor(max_workers=max(workers, len(BEHAVIOURS))) as pool:
        futures = {b: pool.submit(run_behaviour, cfg, b) for b in BEHAVIOURS}
        return {b: f.result() for b, f in futures.items()}


def verdict(res: dict) -> tuple[str, str]:
    walk, jerky = res["scripted gait"], res["jerky gait"]
    noisy, stand, burst = res["noisy gait"], res["stand still"], res["burst gait"]
    if walk["distance"] < 0.5:
        return "INCONCLUSIVE", (f"the scripted walk covered only {walk['distance']:.0%} of the "
                                "asked distance on these physics, so this cannot tell whether "
                                "the reward prefers walking")
    if walk["reward"] <= stand["reward"]:
        return "FAIL", (f"standing still scores {stand['reward'] - walk['reward']:+.3f}/step "
                        "more than walking; training will find standing")
    notes = []
    if noisy["reward"] >= walk["reward"]:
        notes.append("the noisy walk scores at least as much as the smooth walk")
    if jerky["reward"] >= walk["reward"]:
        notes.append("the jerky walk scores at least as much as the smooth walk")
    if burst["reward"] >= walk["reward"]:
        notes.append("walking in bursts with pauses scores at least as much as walking steadily")
    if notes:
        return "WARN", "; ".join(notes)
    return "PASS", (f"walk beats standing by {walk['reward'] - stand['reward']:+.3f}/step, "
                    "and beats the jerky, noisy and burst walks")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reward", default=rewards.DEFAULT_PRESET, choices=sorted(rewards.PRESETS))
    p.add_argument("--compare", nargs="+", choices=sorted(rewards.PRESETS),
                   help="score several presets side by side (--set applies to each)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="any RigConfig field, as in train.py (repeatable)")
    p.add_argument("--rail-friction", type=float, default=3.0, help="N (rig.xml: 0.2)")
    p.add_argument("--cmd", type=float, default=0.07, help="commanded speed, m/s")
    p.add_argument("--period", type=float, default=4.0, help="s per back-and-forth pass")
    p.add_argument("--steps", type=int, default=600, help="control steps (20 ms each)")
    p.add_argument("--terms", action="store_true", help="print each term's weighted mean")
    args = p.parse_args()

    worst = 0
    for preset in args.compare or [args.reward]:
        cfg = make_config(preset, args.set, args.rail_friction, args.cmd, args.period, args.steps)
        res = score(cfg)
        extra = f", --set {' '.join(args.set)}" if args.set else ""
        print(f"\nreward {preset}{extra}  (rail {args.rail_friction} N, cmd {args.cmd} m/s, "
              f"{args.period} s passes, {args.steps} steps, costs at full strength)")
        print(f"  {'behaviour':14s} {'reward/step':>11s} {'distance/asked':>15s} {'lifts':>6s} {'failed':>7s}")
        for b in BEHAVIOURS:
            r = res[b]
            print(f"  {b:14s} {r['reward']:+11.3f} {r['distance']:15.2f} {r['lifts']:6d} {r['failed']:7d}")
        if args.terms:
            names = list(res["scripted gait"]["terms"])
            print("  weighted terms per step: " + "  ".join(f"{n:>9s}" for n in names))
            for b in BEHAVIOURS:
                print(f"  {b:24s}" + "  ".join(f"{res[b]['terms'][n]:+9.3f}" for n in names))
        label, why = verdict(res)
        print(f"  {label}: {why}")
        worst = max(worst, int(label == "FAIL"))
    sys.exit(worst)


if __name__ == "__main__":
    main()
