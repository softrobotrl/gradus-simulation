"""PPO training for the tip-up tendon-leg tracer.

Run::

    python -m mujoco_leg.train --minutes 55

The generated model must exist first::

    python -m mujoco_to_mjcf.build --out urdf/gradus/tracer --tip-up
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mujoco_leg.config import Config
from mujoco_leg.env import TracerEnv
from mujoco_leg.ppo import (
    adam_init,
    compute_gae,
    init_params,
    make_update,
    policy_step,
    value_fn,
)


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-envs", type=int, default=cfg.n_envs)
    p.add_argument("--n-steps", type=int, default=cfg.n_steps)
    p.add_argument("--n-iters", type=int, default=cfg.n_iters)
    p.add_argument("--minutes", type=float, default=55.0,
                   help="wall-clock training budget")
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--run-name", type=str, default=cfg.run_name)
    p.add_argument("--model", type=str, default=cfg.model_path)
    args = p.parse_args()
    return dataclasses.replace(
        cfg,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_iters=args.n_iters,
        lr=args.lr,
        seed=args.seed,
        run_name=args.run_name,
        model_path=args.model,
        minutes=args.minutes,
    )


def save_checkpoint(path: Path, params, cfg: Config) -> None:
    payload = {
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x), params),
        "config": dataclasses.asdict(cfg),
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


def main() -> None:
    cfg = parse_args()
    run_name = cfg.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out = Path(cfg.log_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    print(f"* run: {out}")
    print(f"* model: {cfg.model_path}")

    env = TracerEnv(cfg)
    print(f"* env: obs_dim={env.obs_dim} act_dim={env.nu} dt={env.dt:.4f}s")

    key = jax.random.PRNGKey(cfg.seed)
    key, k_init, k_reset = jax.random.split(key, 3)
    params, apply = init_params(k_init, env.obs_dim, env.nu, cfg.hidden)
    opt_state = adam_init(params)

    keys = jax.random.split(k_reset, cfg.n_envs)

    @jax.jit
    def rollout(params, states, key):
        def body(carry, _):
            states, key = carry
            k, key = jax.random.split(key)
            obs = env.vobs(states)
            action, logp, value, _ = policy_step(params, apply, obs, k)
            states, _, reward, done, info = env.vstep(states, action)
            return (states, key), (obs, action, logp, value, reward, done, info["err"])

        (states, key), traj = jax.lax.scan(body, (states, key), None, length=cfg.n_steps)
        last_value = value_fn(params, apply, env.vobs(states))
        return states, traj, last_value

    @jax.jit
    def gae(traj, last_value):
        _, _, _, value, reward, done, _ = traj
        return compute_gae(reward, done, value, last_value, cfg.gamma, cfg.lam)

    states, _ = env.vreset(keys)
    update_minibatch = make_update(apply, cfg)

    n = cfg.n_steps * cfg.n_envs
    n_mb = cfg.minibatches
    mb = n // n_mb
    n = mb * n_mb

    csv = out / "metrics.csv"
    csv.write_text("iter,steps,return,err,track,pg,vl,ent,kl,clip,secs\n")

    start = time.time()
    best = -1e9
    for it in range(1, cfg.n_iters + 1):
        t0 = time.time()
        states, traj, last_value = rollout(params, states, key)
        key, k_gae = jax.random.split(key)
        adv, ret = gae(traj, last_value)

        obs, action, logp, value, reward, done, err = traj
        flat = (
            obs.reshape(-1, obs.shape[-1])[:n],
            action.reshape(-1, action.shape[-1])[:n],
            logp.reshape(-1)[:n],
            adv.reshape(-1)[:n],
            ret.reshape(-1)[:n],
        )
        metrics = None
        for _ in range(cfg.epochs):
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n)
            obs_s = flat[0][perm]
            act_s = flat[1][perm]
            logp_s = flat[2][perm]
            adv_s = flat[3][perm]
            ret_s = flat[4][perm]
            for j in range(n_mb):
                sl = slice(j * mb, (j + 1) * mb)
                batch = (obs_s[sl], act_s[sl], logp_s[sl], adv_s[sl], ret_s[sl])
                opt_state, metrics = update_minibatch(opt_state, batch)
            if float(np.asarray(metrics["kl"])) > 0.08:
                break
        params = opt_state["p"]

        jax.block_until_ready((params, metrics))
        ret_mean = float(np.asarray(reward).mean())
        err_mean = float(np.asarray(err).mean())
        track_mean = float(np.asarray(jnp.exp(-0.5 * (np.asarray(err) ** 2) / (cfg.track_sigma ** 2))).mean())
        m = {k: float(np.asarray(v)) for k, v in metrics.items()}
        secs = time.time() - start
        steps = it * n
        line = (
            f"{it:4d} {steps/1e6:7.2f}M  return={ret_mean:7.3f}  "
            f"err={err_mean*1000:5.1f}mm  track={track_mean:5.3f}  "
            f"kl={m['kl']:.4f}  ent={m['ent']:.2f}  {secs:5.0f}s"
        )
        print(line, flush=True)
        with open(csv, "a") as f:
            f.write(
                f"{it},{steps},{ret_mean:.6f},{err_mean:.6f},{track_mean:.6f},"
                f"{m['pg']:.6f},{m['vl']:.6f},{m['ent']:.6f},{m['kl']:.6f},"
                f"{m['clip']:.6f},{secs:.1f}\n"
            )

        if track_mean > best:
            best = track_mean
            save_checkpoint(out / "best.pkl", params, cfg)
        if it % cfg.save_every == 0:
            save_checkpoint(out / "latest.pkl", params, cfg)

        if time.time() - start > cfg.minutes * 60.0:
            print("* wall-clock budget reached")
            break

    save_checkpoint(out / "final.pkl", params, cfg)
    print(f"* done: {out/'final.pkl'}  best_track={best:.3f}")


if __name__ == "__main__":
    main()
