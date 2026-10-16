"""Collect surrogate dataset with a trained PPO policy as the expert.

Paper Sec 3.3 split: mostly random actions from fixed s0 + ~2% expert
trajectories injected (random alone never finds the walking manifold).

Uses GenesisGo2Adapter from collect.py for env + canonical 10D state
[euler_rad(3), lin_vel(3), fn(4)], and the PPO checkpoint in logs/<exp>/
as expert_fn. Output format is identical to collect.py (.npz + _meta.json)
so surrogate.py trains unchanged.

Usage (after go2_train.py finishes):
  python collect_go2.py --exp go2-walking --ckpt 100 --n-random 245 --n-expert 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from surrogate.collect import DatasetMeta, GenesisGo2Adapter, save_dataset


def load_policy(exp: str, ckpt: int, env):
    import pickle
    from rsl_rl.runners import OnPolicyRunner
    import genesis as gs

    log_dir = f"logs/{exp}"
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    return runner.get_inference_policy(device=gs.device)


def collect_go2(exp: str, ckpt: int, n_random: int, n_expert: int, seq_len: int,
                num_envs: int, seed: int, out: str):
    adapter = GenesisGo2Adapter(num_envs=num_envs, seq_len=seq_len, seed=seed)
    E, sd, ad = num_envs, adapter.state_dim, adapter.action_dim
    rng = np.random.default_rng(seed)
    try:
        policy = load_policy(exp, ckpt, adapter.env)
        S = n_random + n_expert
        s_out = np.zeros((S, seq_len, sd))
        a_out = np.zeros((S, seq_len, ad))
        sn_out = np.zeros((S, seq_len, sd))
        # Row layout: first n_expert rows are expert, rest random. Order is
        # irrelevant for training (val split is random by sequence).
        w_e, w_r = 0, n_expert
        with torch.no_grad():
            while (w_r - n_expert) < n_random or w_e < n_expert:
                s0 = adapter.reset_batch()
                take_e = min(E, n_expert - w_e)
                take_r = min(E - take_e, n_random - (w_r - n_expert))
                n_take = take_e + take_r
                if n_take == 0:
                    break
                st = s0[:n_take]
                for t in range(seq_len):
                    at = rng.uniform(-1, 1, size=(n_take, ad)).astype(np.float32)
                    if take_e:
                        obs = adapter.env.get_observations()
                        act = policy(obs)  # (E, 12)
                        at[:take_e] = act[:take_e].cpu().numpy()
                    full = np.zeros((E, ad), dtype=np.float32)
                    full[:n_take] = at
                    sn, ok = adapter.step_batch(full)
                    if not bool(np.all(ok[:n_take])):
                        print("[collect_go2] WARNING: solver error, truncating batch.")
                        break
                    if take_e:
                        s_out[w_e:w_e + take_e, t] = st[:take_e]
                        a_out[w_e:w_e + take_e, t] = at[:take_e]
                        sn_out[w_e:w_e + take_e, t] = sn[:take_e]
                    if take_r:
                        s_out[w_r:w_r + take_r, t] = st[take_e:n_take]
                        a_out[w_r:w_r + take_r, t] = at[take_e:]
                        sn_out[w_r:w_r + take_r, t] = sn[take_e:n_take]
                    st = sn[:n_take]
                w_e += take_e
                w_r += take_r
                print(f"[collect_go2] random {w_r - n_expert}/{n_random} expert {w_e}/{n_expert}")
        save_dataset(out, s_out, a_out, sn_out, DatasetMeta(
            sim=adapter.sim_name, dog=adapter.dog_name, dt=adapter.dt,
            state_dim=sd, action_dim=ad, n_seq=S, seq_len=seq_len,
            expert_frac=n_expert / S, seed=seed,
            extra={"exp": exp, "ckpt": ckpt}))
    finally:
        adapter.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="go2-walking")
    ap.add_argument("--ckpt", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=245)
    ap.add_argument("--n-expert", type=int, default=5, help="~2% of 250 like paper Sec 3.3")
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/go2_genesis_ppo.npz")
    args = ap.parse_args()
    collect_go2(args.exp, args.ckpt, args.n_random, args.n_expert,
                args.seq_len, args.num_envs, args.seed, args.out)


if __name__ == "__main__":
    main()
