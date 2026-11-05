"""CPU MuJoCo vectorised quadruped env for rsl-rl PPO.

Sixteen actions: eight tendon pulls for diagonal group A (FL/BR) and eight for
group B (FR/BL).  A simple box trunk, four foot spheres, flat floor.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

LEGS = ("fl", "fr", "bl", "br")
TIP_NAMES = tuple(f"{p}_tip" for p in LEGS)


@dataclass
class EnvConfig:
    model_path: str = "urdf/gradus/walk/quadruped_phys.xml"
    num_envs: int = 64
    n_substeps: int = 5
    control_dt: float = 0.01
    episode_steps: int = 400
    period: float = 0.5
    stand_height: float = 0.6525
    cmd_max: float = 0.25
    lift_height: float = 0.045
    stance_height: float = 0.015
    action_dim: int = 16
    num_threads: int | None = None
    seed: int = 0


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class QuadrupedCpuEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = asdict(cfg)
        self.device = "cpu"
        self.num_envs = cfg.num_envs
        self.num_actions = cfg.action_dim
        self.max_episode_length = cfg.episode_steps
        self.n_substeps = cfg.n_substeps
        self.dt = cfg.control_dt
        self.rng = np.random.default_rng(cfg.seed)

        m = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.m = m
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.l0 = m.actuator_ctrlrange[:, 1].copy()
        self.stroke = (m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0]).copy()
        self.fmax = (-m.actuator_forcerange[:, 0]).copy()
        self.tip_ids = np.array([m.site(n).id for n in TIP_NAMES])
        self.trunk_id = m.body("trunk").id
        self.qpos0 = m.qpos0.copy()

        self.datas = [mujoco.MjData(m) for _ in range(cfg.num_envs)]
        self.t = np.zeros(cfg.num_envs, dtype=np.int32)
        self.cmd = np.zeros((cfg.num_envs, 3), dtype=np.float64)
        self.phase = np.zeros(cfg.num_envs, dtype=np.float64)
        self.prev_action = np.zeros((cfg.num_envs, cfg.action_dim), dtype=np.float64)
        self.prev_foot_z = np.zeros((cfg.num_envs, 4), dtype=np.float64)

        obs0 = self._obs_one(0)
        self.num_obs = int(obs0.shape[0])
        self.obs = np.zeros((cfg.num_envs, self.num_obs), dtype=np.float32)

        threads = max(1, min(cfg.num_threads or os.cpu_count() or 1, cfg.num_envs))
        self.num_threads = threads
        self._chunks = [np.arange(i, cfg.num_envs, threads) for i in range(threads)]
        self._pool = ThreadPoolExecutor(threads)

        self.reset()

    # ------------------------------------------------------------------ #
    def _reset_one(self, i: int, randomize: bool = True) -> None:
        d = self.datas[i]
        d.qpos[:] = self.qpos0
        d.qvel[:] = 0.0
        d.ctrl[:] = self.l0
        mujoco.mj_forward(self.m, d)
        self.t[i] = 0
        self.phase[i] = 0.0
        self.prev_action[i] = 0.0
        self.cmd[i] = [self.rng.uniform(0.0, self.cfg["cmd_max"]) if randomize else 0.0, 0.0, 0.0]
        self.prev_foot_z[i] = d.site_xpos[self.tip_ids, 2]

    def _obs_one(self, i: int) -> np.ndarray:
        d = self.datas[i]
        q = d.qpos[3:7]
        R = quat_to_R(q)
        v_world = d.cvel[self.trunk_id][:3]
        v_body = R.T @ v_world
        ang = d.cvel[self.trunk_id][3:6]
        z = d.qpos[2]
        foot_z = d.site_xpos[self.tip_ids, 2]
        foot_vz = (foot_z - self.prev_foot_z[i]) / self.dt
        ten = (d.ten_length - self.l0) / self.stroke
        force = -d.actuator_force / self.fmax
        phase = np.array([math.sin(2 * math.pi * self.phase[i]),
                          math.cos(2 * math.pi * self.phase[i])])
        return np.concatenate([
            q, np.array([(z - self.cfg["stand_height"]) / 0.1]),
            v_body, ang, self.cmd[i], phase,
            ten, force, foot_z / 0.1, foot_vz, self.prev_action[i],
        ])

    # ------------------------------------------------------------------ #
    def _reward_one(self, i: int, action: np.ndarray):
        d = self.datas[i]
        q = d.qpos[3:7]
        R = quat_to_R(q)
        v_world = d.cvel[self.trunk_id][:3]
        vx = (R.T @ v_world)[0]
        z = d.qpos[2]
        up_z = R[2, 2]
        foot_z = d.site_xpos[self.tip_ids, 2]
        swing = self.phase[i] < 0.5
        target_z = np.array([
            self.cfg.get("lift_height", 0.045) if swing else self.cfg.get("stance_height", 0.015),
            self.cfg.get("stance_height", 0.015) if swing else self.cfg.get("lift_height", 0.045),
            self.cfg.get("stance_height", 0.015) if swing else self.cfg.get("lift_height", 0.045),
            self.cfg.get("lift_height", 0.045) if swing else self.cfg.get("stance_height", 0.015),
        ])
        r_vel = math.exp(-0.5 * ((vx - self.cmd[i, 0]) / 0.15) ** 2) - 0.5 * ((vx - self.cmd[i, 0]) / 0.3) ** 2
        r_height = math.exp(-0.5 * ((z - self.cfg["stand_height"]) / 0.04) ** 2)
        r_tilt = -0.5 * (1.0 - up_z) / 0.05
        r_yaw = -0.1 * (d.cvel[self.trunk_id][5] / 0.5) ** 2
        r_lift = -0.5 * np.mean((foot_z - target_z) ** 2) / 0.02 ** 2
        r_ten = -0.001 * np.mean(-d.actuator_force / self.fmax)
        r_smooth = -0.01 * np.mean((action - self.prev_action[i]) ** 2)
        reward = r_vel + 0.5 * r_height + r_tilt + r_yaw + 0.2 * r_lift + r_ten + r_smooth + 0.1
        failed = (z < 0.35) or (z > 1.0) or (up_z < 0.5) or (not np.isfinite(d.qpos).all())
        return float(reward), bool(failed), {"vx": vx, "z": z, "up": up_z}

    # ------------------------------------------------------------------ #
    def _phys_chunk(self, idxs: np.ndarray, ctrl: np.ndarray) -> None:
        for i in idxs:
            d = self.datas[i]
            d.ctrl[:] = ctrl[i]
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.m, d)

    def step(self, actions, auto_reset: bool = True):
        action = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        if action.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"actions shape {action.shape}, expected {(self.num_envs, self.num_actions)}")

        pull = np.clip(0.5 * (action + 1.0), 0.0, 1.0)
        pull32 = np.concatenate([pull[:, 0:8], pull[:, 8:16], pull[:, 8:16], pull[:, 0:8]], axis=1)
        target = self.l0[None, :] - self.stroke[None, :] * pull32
        ctrl = np.clip(0.5 * np.array([d.ctrl for d in self.datas]) + 0.5 * target,
                       self.l0[None, :] - self.stroke[None, :], self.l0[None, :])

        list(self._pool.map(lambda idxs: self._phys_chunk(idxs, ctrl), self._chunks))

        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        failed = np.zeros(self.num_envs, dtype=bool)
        final_obs = np.zeros_like(self.obs)
        log = {"vx": np.zeros(self.num_envs), "z": np.zeros(self.num_envs), "up": np.zeros(self.num_envs)}

        for i in range(self.num_envs):
            self.t[i] += 1
            r, fail, info = self._reward_one(i, action[i])
            reward[i] = r
            failed[i] = fail
            done[i] = fail or (self.t[i] >= self.max_episode_length)
            log["vx"][i] = info["vx"]
            log["z"][i] = info["z"]
            log["up"][i] = info["up"]
            self.prev_action[i] = action[i]
            self.prev_foot_z[i] = self.datas[i].site_xpos[self.tip_ids, 2]
            self.phase[i] = (self.phase[i] + self.dt / self.cfg["period"]) % 1.0
            self.obs[i] = self._obs_one(i)
            final_obs[i] = self.obs[i]

        time_out = done & ~failed
        if auto_reset and done.any():
            for i in np.flatnonzero(done):
                self._reset_one(i, randomize=True)
                self.obs[i] = self._obs_one(i)

        self.obs = np.nan_to_num(self.obs, nan=0.0, posinf=0.0, neginf=0.0)
        extras = {
            "time_outs": torch.from_numpy(time_out.astype(np.float32)),
            "final_observation": torch.from_numpy(np.nan_to_num(final_obs, nan=0.0, posinf=0.0, neginf=0.0)),
            "log": {
                "/reward_mean": float(reward.mean()),
                "/vx_mean": float(log["vx"].mean()),
                "/height_mean": float(log["z"].mean()),
                "/up_mean": float(log["up"].mean()),
                "/failed": float(failed.mean()),
            },
        }
        return (
            TensorDict({"policy": torch.from_numpy(self.obs)}, batch_size=[self.num_envs]),
            torch.from_numpy(reward),
            torch.from_numpy(done.astype(np.float32)),
            extras,
        )

    # ------------------------------------------------------------------ #
    def reset(self):
        for i in range(self.num_envs):
            self._reset_one(i, randomize=True)
            self.obs[i] = self._obs_one(i)
        self.obs = np.nan_to_num(self.obs, nan=0.0, posinf=0.0, neginf=0.0)
        return TensorDict({"policy": torch.from_numpy(self.obs)}, batch_size=[self.num_envs])

    def get_observations(self):
        return TensorDict({"policy": torch.from_numpy(self.obs)}, batch_size=[self.num_envs])

    @property
    def episode_length_buf(self):
        return torch.from_numpy(self.t.copy())

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.t[:] = value.detach().cpu().numpy().astype(np.int32)

    def close(self):
        self._pool.shutdown(wait=True)
