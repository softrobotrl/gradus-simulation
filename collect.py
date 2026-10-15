"""Collect (s_t, a_t, s_{t+1}) logs for training surrogate.py.

Paper Sec 3.3: dataset D from the slow high-fidelity sim with random actions
from a fixed s0 + ~2% expert-gait injection. D feeds Surrogate.train().

Works with Genesis now, MuJoCo later: add one Adapter class per sim and
register it in SIM_REGISTRY. File format is sim-agnostic (.npz + _meta.json).

Genesis layout
  go2_env.Go2Env buffers -> canonical paper state (physical units):
    s_t = [base_euler_rad(3), base_lin_vel(3), foot_contact(4)] = 10D
    a_t = raw 12D in [-1, 1] (Go2Env scales by action_scale internally)
  NOTE: Go2Env exposes no foot contacts yet -> contacts are zeros with a
  warning. Wire a real contact_fn in GenesisGo2Adapter once available.

Usage
  python collect.py --sim genesis --n-seq 250 --seq-len 100 --num-envs 64
  python collect.py --sim genesis --n-seq 4 --seq-len 8 --num-envs 2 --out data/smoke.npz  # smoke test
  python collect.py --sim genesis --backend cpu --n-seq 20 --seq-len 100 --num-envs 4  # Apple Metal fallback
"""
from __future__ import annotations

import argparse
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Adapter interface: one class per sim. MuJoCo = implement this later.
# --------------------------------------------------------------------------- #
class DataAdapter(ABC):
    sim_name: str = "unknown"
    dog_name: str = "unknown"

    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def dt(self) -> float: ...

    @property
    def num_envs(self) -> int:
        return getattr(self, "_E", 1)

    @abstractmethod
    def reset_batch(self) -> np.ndarray:
        """Reset all envs to s0. Returns s0 (E, state_dim)."""

    @abstractmethod
    def step_batch(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Step all envs. Returns (s_next (E, state_dim), ok (E,) bool). ok=False = solver blew up, drop."""

    @abstractmethod
    def close(self): ...


# --------------------------------------------------------------------------- #
# Genesis adapter (works now)
# --------------------------------------------------------------------------- #
class GenesisGo2Adapter(DataAdapter):
    sim_name = "genesis"
    dog_name = "go2"

    def __init__(self, num_envs: int, seq_len: int, seed: int = 0,
                 contact_fn: Optional[Callable[[], np.ndarray]] = None,
                 show_viewer: bool = False, backend: str = "gpu"):
        import genesis as gs
        from go2_train import get_cfgs
        from go2_env import Go2Env

        self._gs = gs
        self.contact_fn = contact_fn
        if contact_fn is None:
            print("[collect] WARNING: Go2Env has no foot contacts; fn=zeros(4). Wire contact_fn when available.")

        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        # No mid-batch resets: fixed-length sequences from s0, falls included (valid dynamics).
        env_cfg["episode_length_s"] = 1e6
        env_cfg["termination_if_roll_greater_than"] = 1e6
        env_cfg["termination_if_pitch_greater_than"] = 1e6
        self._env_cfg = env_cfg

        if not gs._initialized:
            gs.init(backend=getattr(gs, backend), precision="32", logging_level="warning", seed=seed)
        self.env = Go2Env(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                          reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=show_viewer)
        self._E = num_envs

    @property
    def state_dim(self) -> int:
        return 10

    @property
    def action_dim(self) -> int:
        return int(self.env.num_actions)

    @property
    def dt(self) -> float:
        return float(self.env.dt)

    def _canonical_state(self) -> np.ndarray:
        import torch
        e = self.env
        euler_rad = e.base_euler.cpu().numpy() * (math.pi / 180.0)  # Go2Env stores degrees
        lin_vel = e.base_lin_vel.cpu().numpy()
        if self.contact_fn is not None:
            fn = np.asarray(self.contact_fn(), dtype=np.float64)
        else:
            fn = np.zeros((e.num_envs, 4), dtype=np.float64)
        return np.concatenate([euler_rad, lin_vel, fn], axis=1).astype(np.float64)

    def reset_batch(self) -> np.ndarray:
        self.env.reset()
        return self._canonical_state()

    def step_batch(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        a_t = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.env.device)
        self.env.step(a_t)
        ok = (~self.env.reset_buf.cpu().numpy()).astype(bool) \
            if hasattr(self.env.reset_buf, "cpu") else np.ones(self._E, dtype=bool)
        # reset_buf is True only on solver error here (termination disabled), so ok=False -> drop.
        return self._canonical_state(), ok

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# MuJoCo adapter (stub: implement when the MJX/MuJoCo dog env lands)
# --------------------------------------------------------------------------- #
class MujocoAdapter(DataAdapter):
    """Plug-in point. Wrap pipeline.TendonLegEnv or a MuJoCo Go2 env so that:
      reset_batch() -> s0 (E, state_dim), step_batch(a) -> (s_next, ok).
    Canonical state must match Genesis: [euler_rad(3), lin_vel(3), fn(4)].
    """
    sim_name = "mujoco"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MujocoAdapter not wired yet. Wrap your MuJoCo env with "
            "reset_batch/step_batch returning the canonical 10D state, then register it."
        )

    @property
    def state_dim(self) -> int:
        return 10

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def dt(self) -> float:
        raise NotImplementedError

    def reset_batch(self) -> np.ndarray:
        raise NotImplementedError

    def step_batch(self, actions: np.ndarray):
        raise NotImplementedError

    def close(self):
        pass


SIM_REGISTRY: Dict[str, type] = {
    "genesis": GenesisGo2Adapter,
    "mujoco": MujocoAdapter,
}


# --------------------------------------------------------------------------- #
# Collection (sim-agnostic)
# --------------------------------------------------------------------------- #
def collect_sequences(adapter: DataAdapter, n_seq: int, seq_len: int,
                      expert_frac: float = 0.02,
                      expert_fn: Optional[Callable[[int, int], np.ndarray]] = None,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out fixed-length sequences from s0. Returns s, a, s_next as (S, T, D)."""
    rng = np.random.default_rng(seed)
    if expert_frac > 0 and expert_fn is None:
        print(f"[collect] expert_frac={expert_frac} but no expert_fn -> all-random.")
        expert_frac = 0.0
    E = adapter.num_envs
    S, T, sd, ad = n_seq, seq_len, adapter.state_dim, adapter.action_dim
    s_out = np.zeros((S, T, sd))
    a_out = np.zeros((S, T, ad))
    sn_out = np.zeros((S, T, sd))
    done = 0
    while done < S:
        s0 = adapter.reset_batch()  # (E, sd)
        n_take = min(E, S - done)
        st = s0[:n_take]
        is_expert = rng.random(n_take) < expert_frac
        for t in range(T):
            at = rng.uniform(-1, 1, size=(n_take, ad))
            for i in np.where(is_expert)[0]:
                at[i] = expert_fn(done + int(i), t)
            s_out[done:done + n_take, t] = st
            a_out[done:done + n_take, t] = at
            full = np.zeros((E, ad))
            full[:n_take] = at
            sn, ok = adapter.step_batch(full)
            sn_out[done:done + n_take, t] = sn[:n_take]
            if not bool(np.all(ok[:n_take])):
                print("[collect] WARNING: solver error mid-batch, truncating batch.")
                break
            st = sn[:n_take]
        done += n_take
        print(f"[collect] {done}/{S} sequences")
    return s_out, a_out, sn_out


@dataclass
class DatasetMeta:
    sim: str
    dog: str
    dt: float
    state_dim: int
    action_dim: int
    n_seq: int
    seq_len: int
    expert_frac: float
    seed: int


def save_dataset(path: str, s: np.ndarray, a: np.ndarray, s_next: np.ndarray, meta: DatasetMeta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, s=s, a=a, s_next=s_next)
    with open(os.path.splitext(path)[0] + "_meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)
    print(f"[collect] saved {s.shape[0]}x{s.shape[1]} state_dim={s.shape[2]} action_dim={a.shape[2]} -> {path}")


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    d = np.load(path)
    with open(os.path.splitext(path)[0] + "_meta.json") as f:
        meta = json.load(f)
    return d["s"], d["a"], d["s_next"], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="genesis", choices=list(SIM_REGISTRY))
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu",
                    help="Genesis backend; use cpu to avoid Apple Metal zero-copy compatibility errors")
    ap.add_argument("--n-seq", type=int, default=250, help="paper optimum Fig.4")
    ap.add_argument("--seq-len", type=int, default=100, help="steps per sequence (~MaxStep 100)")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--expert-frac", type=float, default=0.02, help="paper Sec 3.3: 2%% expert injection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/go2_genesis.npz")
    args = ap.parse_args()

    cls = SIM_REGISTRY[args.sim]
    adapter_kwargs = dict(num_envs=args.num_envs, seq_len=args.seq_len, seed=args.seed)
    if args.sim == "genesis":
        adapter_kwargs["backend"] = args.backend
    adapter: DataAdapter = cls(**adapter_kwargs)
    try:
        s, a, sn = collect_sequences(adapter, args.n_seq, args.seq_len,
                                     expert_frac=args.expert_frac, seed=args.seed)
        save_dataset(args.out, s, a, sn, DatasetMeta(
            sim=adapter.sim_name, dog=adapter.dog_name, dt=adapter.dt,
            state_dim=adapter.state_dim, action_dim=adapter.action_dim,
            n_seq=args.n_seq, seq_len=args.seq_len,
            expert_frac=args.expert_frac, seed=args.seed))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
