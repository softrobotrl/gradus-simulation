"""CPU MuJoCo vectorised quadruped env for rsl-rl PPO.

Thirty-two actions: one residual pull per tendon.  The nominal gait is a
diagonal trot (FL+BR against FR+BL); the policy can correct every leg
independently.  A simple box trunk, four foot spheres, flat floor.
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
    period: float = 0.8
    stand_height: float = 0.6625
    cmd_min: float = 0.10
    cmd_max: float = 0.25
    vel_weight: float = 2.0
    vel_sigma: float = 0.15
    progress_weight: float = 0.7
    survival_bonus: float = 0.0
    lift_height: float = 0.035
    stance_height: float = 0.015
    lift_weight: float = 0.3
    slip_weight: float = 0.2
    contact_z: float = 0.03
    slip_scale: float = 0.20
    slip_cap: float = 2.0
    lat_weight: float = 1.0
    yaw_weight: float = 1.0
    heading_weight: float = 0.5
    tilt_weight: float = 1.0
    fall_penalty: float = 1.0
    gait_amp: float = 0.7
    gait_mode: str = "diagonal"   # "diagonal" or "bound"
    swing_frac: float = 0.5       # fraction of the cycle each leg is swinging
    stance_amp: float = 0.0       # back-tendon pull during stance
    residual_scale: float = 0.5
    action_dim: int = 32
    reset_noise: float = 0.0      # scales initial-state perturbation
    num_threads: int | None = None
    seed: int = 0


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


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
        self.prev_foot_xy = np.zeros((cfg.num_envs, 4, 2), dtype=np.float64)
        self.prev_pos = np.zeros((cfg.num_envs, 3), dtype=np.float64)

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
        noise = self.cfg["reset_noise"]
        if noise > 0.0:
            d.qpos[0] += self.rng.uniform(-0.02 * noise, 0.02 * noise)
            d.qpos[1] += self.rng.uniform(-0.02 * noise, 0.02 * noise)
            yaw = self.rng.uniform(-0.12 * noise, 0.12 * noise)
            roll = self.rng.uniform(-0.03 * noise, 0.03 * noise)
            pitch = self.rng.uniform(-0.03 * noise, 0.03 * noise)
            q = quat_mul(
                quat_mul(
                    np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
                    np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]),
                ),
                np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]),
            )
            d.qpos[3:7] = quat_mul(d.qpos[3:7], q)
            d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
            d.qvel[:6] = self.rng.uniform(-0.02 * noise, 0.02 * noise, size=6)
            mujoco.mj_forward(self.m, d)
        self.t[i] = 0
        self.phase[i] = 0.0
        self.prev_action[i] = 0.0
        cmd = (self.rng.uniform(self.cfg["cmd_min"], self.cfg["cmd_max"])
               if randomize else 0.0)
        self.cmd[i] = [cmd, 0.0, 0.0]
        self.prev_foot_z[i] = d.site_xpos[self.tip_ids, 2]
        self.prev_foot_xy[i] = d.site_xpos[self.tip_ids, :2]
        self.prev_pos[i] = d.qpos[:3]

    def _obs_one(self, i: int) -> np.ndarray:
        d = self.datas[i]
        q = d.qpos[3:7]
        R = quat_to_R(q)
        v_world = (d.qpos[:3] - self.prev_pos[i]) / self.dt
        v_body = R.T @ v_world
        ang = d.qvel[3:6]
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
    def _leg_phase(self, p: float) -> np.ndarray:
        """Per-leg gait phase: diagonal trot or all-leg bound."""
        if self.cfg["gait_mode"] == "diagonal":
            return np.array([p, (p + 0.5) % 1.0, (p + 0.5) % 1.0, p])
        return np.full(4, p, dtype=np.float64)

    def _reward_one(self, i: int, action: np.ndarray):
        d = self.datas[i]
        q = d.qpos[3:7]
        R = quat_to_R(q)
        v_world = (d.qpos[:3] - self.prev_pos[i]) / self.dt
        v_body = R.T @ v_world
        vx, vy = float(v_body[0]), float(v_body[1])
        z = d.qpos[2]
        up_z = R[2, 2]
        yaw = math.atan2(R[1, 0], R[0, 0])
        wz = float(d.qvel[5])
        foot_z = d.site_xpos[self.tip_ids, 2]
        p_leg = self._leg_phase(float(self.phase[i]))
        swing = p_leg < self.cfg["swing_frac"]
        target_z = np.where(
            swing,
            self.cfg.get("lift_height", 0.035),
            self.cfg.get("stance_height", 0.015),
        )

        cmd = float(self.cmd[i, 0])
        err = vx - cmd
        sigma = self.cfg.get("vel_sigma", 0.10)
        r_vel = math.exp(-0.5 * (err / sigma) ** 2)
        r_progress = (float(np.clip(vx / max(cmd, 1e-3), 0.0, 1.5))
                      if cmd > 1e-3 else 0.0)
        r_height = math.exp(-0.5 * ((z - self.cfg["stand_height"]) / 0.04) ** 2)
        r_tilt = -0.5 * (1.0 - up_z) / 0.1
        r_lat = -0.5 * (vy / 0.10) ** 2
        r_yaw = -0.5 * (wz / 0.5) ** 2
        r_heading = -math.sin(yaw) ** 2
        r_lift = -0.5 * np.mean((foot_z - target_z) ** 2) / 0.02 ** 2

        foot_xy = d.site_xpos[self.tip_ids, :2]
        foot_vxy = (foot_xy - self.prev_foot_xy[i]) / self.dt
        slip = np.linalg.norm(foot_vxy, axis=1)
        stance = ~swing
        contact = foot_z < self.cfg["contact_z"]
        active = stance & contact
        if np.any(active):
            rel = np.clip(slip[active] / self.cfg["slip_scale"], 0.0,
                          self.cfg["slip_cap"])
            r_slip = -float(np.mean(rel ** 2))
        else:
            r_slip = 0.0

        r_ten = -0.001 * np.mean(-d.actuator_force / self.fmax)
        r_smooth = -0.01 * np.mean((action - self.prev_action[i]) ** 2)
        reward = (
            self.cfg["vel_weight"] * r_vel
            + self.cfg["progress_weight"] * r_progress
            + 0.5 * r_height
            + self.cfg["tilt_weight"] * r_tilt
            + self.cfg["lat_weight"] * r_lat
            + self.cfg["yaw_weight"] * r_yaw
            + self.cfg["heading_weight"] * r_heading
            + self.cfg["lift_weight"] * r_lift
            + self.cfg["slip_weight"] * r_slip
            + r_ten
            + r_smooth
            + self.cfg["survival_bonus"]
        )
        failed = (z < 0.35) or (z > 1.0) or (up_z < 0.5) or (not np.isfinite(d.qpos).all())
        if failed:
            reward -= self.cfg["fall_penalty"]
        info = {
            "vx": vx, "vy": vy, "z": z, "up": up_z, "yaw": yaw,
            "r_vel": r_vel, "r_progress": r_progress,
            "r_height": r_height, "r_tilt": r_tilt,
            "r_lat": r_lat, "r_yaw": r_yaw, "r_heading": r_heading,
            "r_lift": r_lift, "r_slip": r_slip,
            "r_ten": r_ten, "r_smooth": r_smooth,
        }
        return float(reward), bool(failed), info

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

        # diagonal trot: the swing diagonal pulls its back tendons to lift and
        # drive the foot forward; the stance diagonal can also pull its back
        # tendons (stance_amp) to push the body forward and keep the trunk up.
        # Pulling front and back together cancels propulsion because both
        # directions shorten the leg.
        p = self.phase[:, None]
        if self.cfg["gait_mode"] == "diagonal":
            p_leg = np.concatenate([p, (p + 0.5) % 1.0, (p + 0.5) % 1.0, p], axis=1)
        else:
            p_leg = np.repeat(p, 4, axis=1)
        swing_frac = float(self.cfg["swing_frac"])
        swing = p_leg < swing_frac
        back = np.where(
            swing,
            self.cfg["gait_amp"] * np.sin(np.pi * p_leg / max(swing_frac, 1e-6)),
            0.0,
        )
        stance = np.zeros_like(back)
        if self.cfg["stance_amp"] > 0.0 and swing_frac < 1.0:
            stance = np.where(
                ~swing,
                self.cfg["stance_amp"]
                * np.sin(np.pi * (p_leg - swing_frac) / max(1.0 - swing_frac, 1e-6)),
                0.0,
            )
        pull = np.clip(back + stance, 0.0, 1.0)
        nom = np.zeros((self.num_envs, 4, 8), dtype=np.float64)
        nom[:, :, 2] = pull       # inner_back
        nom[:, :, 6] = pull       # outer_back
        nom32 = nom.reshape(self.num_envs, 32)

        # policy action is a residual on the nominal gait.  32 actions map
        # directly to the 32 tendons; 16 actions keep the old diagonal-pair
        # mapping for checkpoint compatibility.
        if self.num_actions == 16:
            res32 = np.concatenate(
                [action[:, 0:8], action[:, 8:16], action[:, 8:16], action[:, 0:8]], axis=1
            )
        else:
            res32 = action
        pull32 = np.clip(nom32 + self.cfg["residual_scale"] * res32, 0.0, 1.0)
        target = self.l0[None, :] - self.stroke[None, :] * pull32
        ctrl = np.clip(0.5 * np.array([d.ctrl for d in self.datas]) + 0.5 * target,
                       self.l0[None, :] - self.stroke[None, :], self.l0[None, :])

        list(self._pool.map(lambda idxs: self._phys_chunk(idxs, ctrl), self._chunks))

        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        failed = np.zeros(self.num_envs, dtype=bool)
        final_obs = np.zeros_like(self.obs)
        term_keys = (
            "r_vel", "r_progress", "r_height", "r_tilt", "r_lat", "r_yaw",
            "r_heading", "r_lift", "r_slip", "r_ten", "r_smooth",
        )
        log = {k: np.zeros(self.num_envs) for k in ("vx", "vy", "z", "up", "yaw") + term_keys}

        for i in range(self.num_envs):
            self.t[i] += 1
            r, fail, info = self._reward_one(i, action[i])
            reward[i] = r
            failed[i] = fail
            done[i] = fail or (self.t[i] >= self.max_episode_length)
            log["vx"][i] = info["vx"]
            log["vy"][i] = info["vy"]
            log["z"][i] = info["z"]
            log["up"][i] = info["up"]
            log["yaw"][i] = info["yaw"]
            for k in term_keys:
                log[k][i] = info[k]
            self.prev_action[i] = action[i]
            self.prev_foot_z[i] = self.datas[i].site_xpos[self.tip_ids, 2]
            self.prev_foot_xy[i] = self.datas[i].site_xpos[self.tip_ids, :2]
            self.prev_pos[i] = self.datas[i].qpos[:3]
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
                "/vy_mean": float(log["vy"].mean()),
                "/height_mean": float(log["z"].mean()),
                "/up_mean": float(log["up"].mean()),
                "/yaw_mean": float(log["yaw"].mean()),
                "/failed": float(failed.mean()),
                **{f"/{k}": float(log[k].mean()) for k in term_keys},
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
