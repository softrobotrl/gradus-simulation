"""DNN surrogate world model: f_hat(s_t, a_t) -> s_{t+1}.

Paper replication (Niu et al, SoftQ MBRL, Sec 3.3):
  - Surrogate approximates the slow high-fidelity sim (SimScape in paper,
    MuJoCo/Genesis here) so SAC can train fast, then post-train (PT) on real sim.
  - Loss Eq.4: mean 0.5*||s' - f_hat(s,a)||^2.
  - Metrics Eq.5-8: R, NRMSE (1-step) + R_T, NRMSE_T (T-step open-loop rollout).
  - Paper arch TABLE 1: 14 -> 64 -> 128 -> 64 (ReLU) -> 10.
    Here dims are free: input_dim = state_dim + action_dim, output_dim = state_dim.

Public API is canonical physical units in/out. Standardization lives INSIDE
the class (fit on train data, saved with checkpoint), so callers and
sim adapters never handle scaling manually.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Scaler (internal, saved with checkpoint)
# --------------------------------------------------------------------------- #
class StandardScaler:
    """Per-feature z-score. Fit on train only."""

    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + self.eps
        # Constant features -> std ~ eps, keep them at 0 after centering.
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        s = cls(eps=d.get("eps", 1e-8))
        s.mean = np.array(d["mean"], dtype=np.float64)
        s.std = np.array(d["std"], dtype=np.float64)
        return s


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (64, 128, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SurrogateConfig:
    state_dim: int
    action_dim: int
    hidden_dims: Tuple[int, ...] = (64, 128, 64)
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 100
    val_frac: float = 0.1
    patience: int = 10
    seed: int = 0
    # provenance, stored in checkpoint so reload is safe
    sim: str = "unknown"
    dog: str = "unknown"
    dt: float = 0.02
    extra: Dict = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.action_dim


# --------------------------------------------------------------------------- #
# Surrogate wrapper: train() on sim logs, predict() next state
# --------------------------------------------------------------------------- #
class Surrogate:
    def __init__(self, config: SurrogateConfig, device: Optional[str] = None):
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SurrogateMLP(config.input_dim, config.state_dim, config.hidden_dims).to(self.device)
        self.x_scaler = StandardScaler()  # over concat(s, a)
        self.y_scaler = StandardScaler()  # over s'
        self._fitted = False
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # -- data helpers: accept flat (N,D) or sequenced (S,T,D) -- #
    @staticmethod
    def _to_flat(s, a, s_next) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        s_next = np.asarray(s_next, dtype=np.float64)
        if s.ndim == 3:  # (seq, time, dim) -> (N, dim)
            s = s.reshape(-1, s.shape[-1])
            a = a.reshape(-1, a.shape[-1])
            s_next = s_next.reshape(-1, s_next.shape[-1])
        if s.ndim != 2 or a.ndim != 2 or s_next.ndim != 2:
            raise ValueError("Expected flat (N,D) or sequenced (S,T,D) arrays")
        if s.shape != s_next.shape or len(s) != len(a):
            raise ValueError("State, action and next-state arrays must have matching transitions")
        return s, a, s_next

    def _split_by_sequence(
        self, s, a, s_next, val_frac: float
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """Split by sequence idx when input is (S,T,D); else random flat split."""
        if not 0 < val_frac < 1 or len(s) < 2:
            raise ValueError("Validation requires at least two samples/sequences and 0 < val_frac < 1")
        if np.asarray(s).ndim == 3:
            n_seq = len(s)
            idx = np.random.permutation(n_seq)
            n_val = max(1, int(n_seq * val_frac))
            va, tr = idx[:n_val], idx[n_val:]
            return (
                (s[tr].reshape(-1, s.shape[-1]), a[tr].reshape(-1, a.shape[-1]), s_next[tr].reshape(-1, s_next.shape[-1])),
                (s[va].reshape(-1, s.shape[-1]), a[va].reshape(-1, a.shape[-1]), s_next[va].reshape(-1, s_next.shape[-1])),
            )
        n = len(s)
        idx = np.random.permutation(n)
        n_val = max(1, int(n * val_frac))
        va, tr = idx[:n_val], idx[n_val:]
        return ((s[tr], a[tr], s_next[tr]), (s[va], a[va], s_next[va]))

    # -- train: Eq.4 -- #
    def train(self, s, a, s_next, verbose: bool = True) -> Dict[str, List[float]]:
        s, a, s_next = (np.asarray(x, dtype=np.float64) for x in (s, a, s_next))
        flat_s, flat_a, _ = self._to_flat(s, a, s_next)
        assert flat_s.shape[1] == self.cfg.state_dim, f"s dim {flat_s.shape[1]} != {self.cfg.state_dim}"
        assert flat_a.shape[1] == self.cfg.action_dim, f"a dim {flat_a.shape[1]} != {self.cfg.action_dim}"
        if self.cfg.epochs < 1:
            raise ValueError("epochs must be positive")
        (tr_s, tr_a, tr_n), (va_s, va_a, va_n) = self._split_by_sequence(
            np.asarray(s), np.asarray(a), np.asarray(s_next), self.cfg.val_frac
        )
        x_tr = np.concatenate([tr_s, tr_a], axis=1)
        x_va = np.concatenate([va_s, va_a], axis=1)
        self.x_scaler.fit(x_tr)
        self.y_scaler.fit(tr_n)
        self._fitted = True

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

        x_tr_t = t(self.x_scaler.transform(x_tr))
        y_tr_t = t(self.y_scaler.transform(tr_n))
        x_va_t = t(self.x_scaler.transform(x_va))
        y_va_t = t(self.y_scaler.transform(va_n))

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()  # Eq.4 up to the constant 0.5 factor
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tr_t, y_tr_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        hist = {"train": [], "val": []}
        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.cfg.epochs):
            self.model.train()
            tot = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            tr_loss = tot / len(loader.dataset)
            self.model.eval()
            with torch.no_grad():
                va_loss = loss_fn(self.model(x_va_t), y_va_t).item()
            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)
            if verbose and (ep % 10 == 0 or ep == self.cfg.epochs - 1):
                print(f"[surrogate] ep {ep:3d} train {tr_loss:.5f} val {va_loss:.5f}")
            if va_loss < best - 1e-6:
                best, best_state, bad = va_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    if verbose:
                        print(f"[surrogate] early stop at ep {ep}, best val {best:.5f}")
                    break
        self.model.load_state_dict(best_state)
        return hist

    # -- inference: canonical units in/out -- #
    @torch.no_grad()
    def predict(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        assert self._fitted, "call train() or load() first"
        s = np.asarray(s, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        single = s.ndim == 1
        if single:
            s, a = s[None, :], a[None, :]
        x = np.concatenate([s, a], axis=1)
        x_t = torch.from_numpy(self.x_scaler.transform(x).astype(np.float32)).to(self.device)
        self.model.eval()
        y = self.model(x_t).cpu().numpy()
        out = self.y_scaler.inverse(y)
        return out[0] if single else out

    @torch.no_grad()
    def rollout(self, s0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """T-step open-loop: s_hat_{t+1} = f_hat(s_hat_t, a_t), s_hat_0 = s0."""
        s0 = np.asarray(s0, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        preds = np.zeros((len(actions), self.cfg.state_dim))
        cur = s0
        for i, at in enumerate(actions):
            cur = self.predict(cur, at)
            preds[i] = cur
        return preds

    # -- metrics Eq.5-8 -- #
    def evaluate(self, s, a, s_next, max_T: Optional[int] = None) -> Dict[str, float]:
        """1-step R/NRMSE + optional T-step rollout NRMSE_T/R_T on sequenced data."""
        s = np.asarray(s, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        s_next = np.asarray(s_next, dtype=np.float64)
        out: Dict[str, float] = {}
        pred_1 = self.predict(s.reshape(-1, s.shape[-1]), a.reshape(-1, a.shape[-1]))
        true_1 = s_next.reshape(-1, s_next.shape[-1])
        out["R"], out["NRMSE"] = _r_nrmse(true_1, pred_1)
        if s.ndim == 3:
            T = s.shape[1] if max_T is None else min(max_T, s.shape[1])
            nrmses, rs = [], []
            for seq in range(s.shape[0]):
                ph = self.rollout(s[seq, 0], a[seq, :T])
                th = s_next[seq, :T]
                r, n = _r_nrmse(th, ph)
                rs.append(r)
                nrmses.append(n)
            out["R_T"], out["NRMSE_T"] = float(np.mean(rs)), float(np.mean(nrmses))
            out["T"] = T
        return out

    # -- persistence: self-contained checkpoint -- #
    def save(self, path: str):
        assert self._fitted, "nothing to save before train()"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.cfg),
                "x_scaler": self.x_scaler.to_dict(),
                "y_scaler": self.y_scaler.to_dict(),
            },
            path,
        )
        print(f"[surrogate] saved -> {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "Surrogate":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = SurrogateConfig(**ckpt["config"])
        obj = cls(cfg, device=device)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.x_scaler = StandardScaler.from_dict(ckpt["x_scaler"])
        obj.y_scaler = StandardScaler.from_dict(ckpt["y_scaler"])
        obj._fitted = True
        return obj


def _r_nrmse(true: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """Paper Eq.5 (R) and Eq.6 (NRMSE), averaged over dims, NaN-safe."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = true.max(axis=0) - true.min(axis=0)
    denom = np.where(denom < 1e-12, 1.0, denom)
    nrmse = float(np.sqrt(np.mean(((true - pred) / denom) ** 2)))
    rs = []
    for j in range(true.shape[1]):
        t, p = true[:, j], pred[:, j]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            continue
        rs.append(float(np.corrcoef(t, p)[0, 1]))
    r = float(np.mean(rs)) if rs else float("nan")
    return r, nrmse
