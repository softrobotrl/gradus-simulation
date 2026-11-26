"""Speed benchmark and exact-behaviour recording for the rig env.

    python -m single_leg_demo.bench time                  # env steps/s (DR on and off) + a short training run
    python -m single_leg_demo.bench record OUT.npz        # record obs/reward/state on fixed seeds
    python -m single_leg_demo.bench compare REF.npz       # re-run and compare with a recording

Use `record` before an optimisation and `compare` after it: a speed-up must
not change what the env does (same seeds and actions -> same numbers).
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from single_leg_demo import rewards
from single_leg_demo.rig_env import RigConfig, SingleLegRigEnv

CASES = {   # name -> RigConfig overrides; short episodes so resets are exercised
    "paired_dr": dict(servo_layout="paired_x", dr_enabled=True),
    "paired_nominal": dict(servo_layout="paired_x", dr_enabled=False),
    "independent_dr": dict(servo_layout="independent", servo_torque=0.0, dr_enabled=True,
                           servo_load_obs=True),
}


def make_env(num_envs: int, threads: int, **over) -> SingleLegRigEnv:
    cfg = RigConfig(num_envs=num_envs, num_threads=threads, rail_friction=3.0, seed=3, **over)
    rewards.apply_preset(cfg, "v5")
    return SingleLegRigEnv(cfg)


def time_env(num_envs: int, threads: int, steps: int, repeats: int, dr: bool) -> float:
    """Median seconds per batched env.step."""
    env = make_env(num_envs, threads, dr_enabled=dr)
    rng = np.random.default_rng(0)
    acts = [torch.from_numpy(rng.uniform(-1, 1, (num_envs, env.num_actions)).astype(np.float32))
            for _ in range(steps)]
    for a in acts[:5]:
        env.step(a)
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        for a in acts:
            env.step(a)
        runs.append((time.perf_counter() - t) / steps)
    env.close()
    return statistics.median(runs)


def time_training(iterations: int, threads: int) -> dict:
    """Run train.py briefly; median collection / learning seconds per iteration (first 2 dropped)."""
    cmd = [sys.executable, "-m", "single_leg_demo.train", "--run-name", "_bench", "--iterations",
           str(iterations), "--num-envs", "64", "--num-threads", str(threads), "--dr",
           "--rail-friction", "3", "--warmup", "0", "--seed", "0"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    col = [float(x) for x in re.findall(r"Collection time:\s*([\d.]+)s", out)][2:]
    lrn = [float(x) for x in re.findall(r"Learning time:\s*([\d.]+)s", out)][2:]
    import shutil
    shutil.rmtree(Path("logs/single_leg_demo/_bench"), ignore_errors=True)
    return {"collection_s": statistics.median(col), "learning_s": statistics.median(lrn)}


def record(steps: int = 60) -> dict:
    out = {}
    for name, over in CASES.items():
        env = make_env(6, 3, episode_steps=25, **over)
        rng = np.random.default_rng(1)
        obs = env.reset()
        rows = {"obs": [obs["policy"].numpy().copy()], "critic": [obs["critic"].numpy().copy()]}
        rew, done, qpos, log = [], [], [], []
        for _ in range(steps):
            a = torch.from_numpy(rng.uniform(-1.2, 1.2, (6, env.num_actions)).astype(np.float32))
            o, r, d, ex = env.step(a)
            rows["obs"].append(o["policy"].numpy().copy())
            rows["critic"].append(o["critic"].numpy().copy())
            rew.append(r.numpy().copy())
            done.append(d.numpy().copy())
            qpos.append(np.stack([dd.qpos.copy() for dd in env.datas]))
            log.append([ex["log"][k] for k in sorted(ex["log"])])
        env.close()
        for k, v in rows.items():
            out[f"{name}/{k}"] = np.stack(v)
        out[f"{name}/reward"] = np.stack(rew)
        out[f"{name}/done"] = np.stack(done)
        out[f"{name}/qpos"] = np.stack(qpos)
        out[f"{name}/log"] = np.array(log)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time")
    t.add_argument("--threads", type=int, default=3)
    t.add_argument("--steps", type=int, default=48)
    t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--train-iterations", type=int, default=8, help="0 skips the training timing")
    r = sub.add_parser("record")
    r.add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    args = p.parse_args()

    if args.cmd == "time":
        n = 64
        for dr in (True, False):
            s = time_env(n, args.threads, args.steps, args.repeats, dr)
            print(f"env.step, {n} envs, {args.threads} threads, DR {'on ' if dr else 'off'}: "
                  f"{s * 1000:7.1f} ms per step  ({n / s:7.0f} env-steps/s)", flush=True)
        if args.train_iterations:
            tr = time_training(args.train_iterations, args.threads)
            print(f"training (64 envs x 24 steps, DR on): collection {tr['collection_s']:.2f} s, "
                  f"learning {tr['learning_s']:.2f} s per iteration")
    elif args.cmd == "record":
        np.savez_compressed(args.out, **record())
        print(f"* recorded {args.out}")
    else:
        ref = np.load(args.ref)
        new = record()
        worst = 0.0
        for k in ref.files:
            a, b = ref[k], new[k]
            if a.shape != b.shape:
                print(f"  {k}: shape {a.shape} -> {b.shape}  MISMATCH")
                worst = np.inf
                continue
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
            worst = max(worst, diff)
            if diff > 0:
                print(f"  {k}: max abs difference {diff:.3g}")
        print("* IDENTICAL" if worst == 0 else f"* worst difference {worst:.3g}")
        sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
