"""CPU MuJoCo env: demo leg on a one-dimensional sliding rig.

The hip is bolted to a carriage with a single prismatic joint along +X.  The
only ground contact is a small sphere at the foot, so the leg has to bend
itself with the eight tendon servos to push the carriage back and forth.

Observations are proprioceptive only (plus the commanded velocity):

* slider position and velocity (the linear-encoder state),
* commanded velocity and the phase of the back-and-forth command,
* 13 ball-joint rotation vectors and velocities,
* normalised tendon lengths, actuator forces, servo targets,
* the previous action.

Actions are eight residuals on the tendon pull that holds the slightly bent
reset pose.  ``pull = pull_bias + residual_scale * action`` in ``[0, 1]``.
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

N_BALL = 13
N_TENDON = 8
FOOT_RADIUS = 0.015


@dataclass
class RigConfig:
    model_path: str = "urdf/gradus/demo/rig.xml"
    num_envs: int = 64
    n_substeps: int = 10
    control_dt: float = 0.02
    episode_steps: int = 400
    rail_half: float = 0.45

    # Desired speed: the command alternates sign every cmd_period seconds.
    cmd_min: float = 0.03
    cmd_max: float = 0.12
    cmd_period: float = 2.0
    cmd_period_jitter: float = 0.20
    cmd_resample: bool = True        # new random magnitude at every reversal
    obs_cmd_scale: float = 0.16      # fixed normaliser, independent of cmd_max

    # Reward weights.
    vel_weight: float = 2.0
    vel_sigma: float = 0.06
    vel_l2_weight: float = 0.02      # keeps a gradient during fast reversals
    progress_weight: float = 0.5
    rail_weight: float = 0.5
    foot_weight: float = 0.3
    lat_weight: float = 0.5
    effort_weight: float = 0.02
    rate_weight: float = 0.02
    tension_weight: float = 0.001
    action_l2_weight: float = 0.005   # keep raw actions out of hard saturation
    failure_penalty: float = 5.0

    # Termination / reset.
    bend_limit: float = 0.90          # rad, per ball joint
    foot_lift_limit: float = 0.12     # m, foot centre above the floor
    reset_bend: float = 0.012         # rad per ball joint, sign randomised
    reset_noise: float = 0.25         # scales reset perturbations
    reset_settle: int = 20            # physics steps after the reset pose

    residual_scale: float = 0.6
    num_threads: int | None = None
    seed: int = 0


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternion (w, x, y, z) -> (..., 3) rotation vector."""
    w = np.abs(q[..., 0])
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, w[..., None])
    axis = v / np.maximum(n, 1e-9)
    return angle * axis


def _bend_angle(q: np.ndarray) -> np.ndarray:
    return 2.0 * np.arccos(np.clip(np.abs(q[..., 0]), 0.0, 1.0))


class SingleLegRigEnv:
    def __init__(self, cfg: RigConfig):
        self.cfg = asdict(cfg)
        self.device = "cpu"
        self.num_envs = cfg.num_envs
        self.num_actions = N_TENDON
        self.max_episode_length = cfg.episode_steps
        self.n_substeps = cfg.n_substeps
        self.dt = cfg.control_dt
        self.rng = np.random.default_rng(cfg.seed)

        m = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.m = m
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        if self.nu != N_TENDON:
            raise RuntimeError(f"expected {N_TENDON} tendon actuators, model has {self.nu}")

        self.l0 = m.actuator_ctrlrange[:, 1].copy()
        self.stroke = (m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0]).copy()
        self.fmax = (-m.actuator_forcerange[:, 0]).copy()
        self.ctrl_lo = self.l0 - self.stroke

        self.slide_joint = m.joint("slide")
        self.slide_qadr = int(self.slide_joint.qposadr[0])
        self.ball_joints = [m.joint(f"{i:02d}") for i in range(N_BALL)]
        self.ball_qadr = np.array([int(j.qposadr[0]) for j in self.ball_joints])
        self.ball_dadr = np.array([int(j.dofadr[0]) for j in self.ball_joints])
        self.tip_id = m.site("tip").id
        self.qpos_straight = m.qpos0.copy()

        self.datas = [mujoco.MjData(m) for _ in range(cfg.num_envs)]
        self.t = np.zeros(cfg.num_envs, dtype=np.int32)
        self.cmd = np.zeros(cfg.num_envs, dtype=np.float64)
        self.cmd_mag = np.zeros(cfg.num_envs, dtype=np.float64)
        self.cmd_sign0 = np.ones(cfg.num_envs, dtype=np.float64)
        self.cmd_period = np.full(cfg.num_envs, cfg.cmd_period, dtype=np.float64)
        self.cmd_segment = np.zeros(cfg.num_envs, dtype=np.int32)
        self.pull_bias = np.zeros((cfg.num_envs, N_TENDON), dtype=np.float64)
        self.prev_action = np.zeros((cfg.num_envs, N_TENDON), dtype=np.float64)

        obs0 = self._obs_one(0)
        self.num_obs = int(obs0.shape[0])
        self.obs = np.zeros((cfg.num_envs, self.num_obs), dtype=np.float32)

        threads = max(1, min(cfg.num_threads or os.cpu_count() or 1, cfg.num_envs))
        self.num_threads = threads
        self._chunks = [np.arange(i, cfg.num_envs, threads) for i in range(threads)]
        self._pool = ThreadPoolExecutor(threads)

        self.reset()

    # ------------------------------------------------------------------ #
    def _set_cmd(self, i: int) -> None:
        t = self.t[i] * self.dt
        period = max(self.cmd_period[i], 1e-3)
        segment = int(math.floor(t / period + 1e-9))
        if self.cfg["cmd_resample"] and segment != self.cmd_segment[i]:
            # A fixed per-episode amplitude lets the policy learn one open-loop
            # pump and ignore the command.  Resampling at every reversal keeps
            # the commanded magnitude informative.
            self.cmd_mag[i] = self.rng.uniform(self.cfg["cmd_min"], self.cfg["cmd_max"])
        self.cmd_segment[i] = segment
        sign = self.cmd_sign0[i] * (1.0 if segment % 2 == 0 else -1.0)
        self.cmd[i] = sign * self.cmd_mag[i]

    def _phase(self, i: int) -> float:
        period = max(self.cmd_period[i], 1e-3)
        return float((self.t[i] * self.dt / period) % 1.0)

    def _obs_one(self, i: int) -> np.ndarray:
        d = self.datas[i]
        quat = np.stack([d.qpos[a:a + 4] for a in self.ball_qadr])
        rotvec = _quat_to_rotvec(quat).reshape(-1)
        qvel_ball = np.concatenate([d.qvel[a:a + 3] for a in self.ball_dadr])
        ten = (self.l0 - d.ten_length) / self.stroke
        force = -d.actuator_force / self.fmax
        ctrl = (self.l0 - d.ctrl) / self.stroke
        phase = self._phase(i)
        return np.concatenate([
            np.array([
                d.qpos[self.slide_qadr] / self.cfg["rail_half"],
                d.qvel[0] / 0.20,
                self.cmd[i] / max(self.cfg["obs_cmd_scale"], 1e-3),
                math.sin(2 * math.pi * phase),
                math.cos(2 * math.pi * phase),
            ]),
            rotvec,
            qvel_ball * 0.05,
            np.clip(ten, -0.5, 1.5),
            np.clip(force, 0.0, 1.5),
            np.clip(ctrl, 0.0, 1.0),
            self.prev_action[i],
        ])

    # ------------------------------------------------------------------ #
    def _reset_one(self, i: int, randomize: bool = True) -> None:
        d = self.datas[i]
        d.qpos[:] = self.qpos_straight
        d.qvel[:] = 0.0
        # Randomise which way the slight initial bend leans, so the policy
        # learns to push in both directions from the first update.
        bend = self.cfg["reset_bend"]
        if randomize and self.rng.random() < 0.5:
            bend = -bend
        half = 0.5 * bend
        bend_q = np.array([math.cos(half), 0.0, math.sin(half), 0.0])
        for adr in self.ball_qadr:
            d.qpos[adr:adr + 4] = bend_q

        noise = self.cfg["reset_noise"] if randomize else 0.0
        if noise > 0.0:
            for adr in self.ball_qadr:
                n = np.array([
                    1.0,
                    self.rng.normal(0.0, 0.003 * noise),
                    self.rng.normal(0.0, 0.003 * noise),
                    self.rng.normal(0.0, 0.003 * noise),
                ])
                n /= np.linalg.norm(n)
                w1, x1, y1, z1 = d.qpos[adr:adr + 4]
                w2, x2, y2, z2 = n
                q = np.array([
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ])
                d.qpos[adr:adr + 4] = q / np.linalg.norm(q)
        mujoco.mj_forward(self.m, d)
        tip = d.site_xpos[self.tip_id]
        # put the foot near the rail centre; the hip moves the other way
        d.qpos[self.slide_qadr] = -tip[0] + self.rng.uniform(-0.03, 0.03) * noise
        d.qvel[0] = self.rng.uniform(-0.02, 0.02) * noise

        d.ctrl[:] = self.l0
        mujoco.mj_forward(self.m, d)
        pull = np.clip((self.l0 - d.ten_length) / self.stroke, 0.0, 1.0)
        self.pull_bias[i] = pull
        d.ctrl[:] = self.l0 - self.stroke * pull

        if self.cfg["reset_settle"] > 0:
            for _ in range(int(self.cfg["reset_settle"])):
                mujoco.mj_step(self.m, d)
            d.ctrl[:] = self.l0 - self.stroke * pull

        self.t[i] = 0
        self.prev_action[i] = 0.0
        self.cmd_mag[i] = (self.rng.uniform(self.cfg["cmd_min"], self.cfg["cmd_max"])
                           if randomize else self.cfg["cmd_min"])
        self.cmd_sign0[i] = 1.0 if self.rng.random() < 0.5 else -1.0
        self.cmd_segment[i] = 0
        jitter = self.cfg["cmd_period_jitter"]
        self.cmd_period[i] = self.cfg["cmd_period"] * (1.0 + self.rng.uniform(-jitter, jitter))
        self._set_cmd(i)

    def _reward_one(self, i: int, action: np.ndarray, action_raw: np.ndarray):
        d = self.datas[i]
        vx = float(d.qvel[0])
        x = float(d.qpos[self.slide_qadr])
        cmd = float(self.cmd[i])
        tip = d.site_xpos[self.tip_id]
        foot_z = float(tip[2])
        foot_y = float(tip[1])
        quat = np.stack([d.qpos[a:a + 4] for a in self.ball_qadr])
        bend = _bend_angle(quat)
        force = -d.actuator_force / self.fmax
        pull = (self.l0 - d.ctrl) / self.stroke

        err = vx - cmd
        sigma = max(self.cfg["vel_sigma"], 1e-6)
        r_vel = math.exp(-0.5 * (err / sigma) ** 2)
        r_vel_l2 = -self.cfg["vel_l2_weight"] * (err / sigma) ** 2
        r_prog = (float(np.clip(vx / cmd, 0.0, 1.5)) if abs(cmd) > 1e-6 else 0.0)
        rail_half = self.cfg["rail_half"]
        over = max(0.0, abs(x) - 0.7 * rail_half)
        r_rail = -0.5 * (over / (0.3 * rail_half)) ** 2
        r_foot = -0.5 * ((foot_z - FOOT_RADIUS) / 0.02) ** 2
        r_lat = -0.5 * (foot_y / 0.03) ** 2
        r_effort = -self.cfg["effort_weight"] * float(np.mean(pull ** 2))
        r_rate = -self.cfg["rate_weight"] * float(np.mean((action - self.prev_action[i]) ** 2))
        r_ten = -self.cfg["tension_weight"] * float(np.mean(force))
        # Penalise the un-clipped action.  Without this the policy drifts into
        # bang-bang actions (|a| > 5), clipping removes all command magnitude
        # information, and the leg pumps at one fixed amplitude.
        r_action_l2 = -self.cfg["action_l2_weight"] * float(np.mean(action_raw ** 2))

        reward = (
            self.cfg["vel_weight"] * r_vel
            + r_vel_l2
            + self.cfg["progress_weight"] * r_prog
            + self.cfg["rail_weight"] * r_rail
            + self.cfg["foot_weight"] * r_foot
            + self.cfg["lat_weight"] * r_lat
            + r_effort + r_rate + r_ten + r_action_l2
        )
        failed = (
            foot_z > self.cfg["foot_lift_limit"]
            or bend.max() > self.cfg["bend_limit"]
            or abs(x) > rail_half * 0.98
            or not np.isfinite(d.qpos).all()
        )
        if failed:
            reward -= self.cfg["failure_penalty"]
        info = {
            "vx": vx, "x": x, "cmd": cmd, "err": abs(err),
            "foot_z": foot_z, "foot_y": foot_y, "bend": float(bend.max()),
            "r_vel": r_vel, "r_vel_l2": r_vel_l2, "r_prog": r_prog, "r_rail": r_rail,
            "r_foot": r_foot, "r_lat": r_lat,
            "r_effort": r_effort, "r_rate": r_rate, "r_ten": r_ten,
            "r_action_l2": r_action_l2,
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
        action_raw = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_raw = action_raw.astype(np.float64)
        action = np.clip(action_raw, -1.0, 1.0)
        if action.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"actions shape {action.shape}, expected {(self.num_envs, self.num_actions)}")

        target_pull = np.clip(
            self.pull_bias + self.cfg["residual_scale"] * action, 0.0, 1.0)
        target_ctrl = self.l0[None, :] - self.stroke[None, :] * target_pull
        ctrl_now = np.array([d.ctrl for d in self.datas])
        ctrl = np.clip(0.5 * ctrl_now + 0.5 * target_ctrl, self.ctrl_lo, self.l0)
        list(self._pool.map(lambda idxs: self._phys_chunk(idxs, ctrl), self._chunks))

        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        failed = np.zeros(self.num_envs, dtype=bool)
        final_obs = np.zeros_like(self.obs)
        keys = ("vx", "x", "cmd", "err", "foot_z", "foot_y", "bend",
                "r_vel", "r_vel_l2", "r_prog", "r_rail", "r_foot", "r_lat",
                "r_effort", "r_rate", "r_ten", "r_action_l2")
        log = {k: np.zeros(self.num_envs) for k in keys}

        for i in range(self.num_envs):
            self.t[i] += 1
            self._set_cmd(i)
            r, fail, info = self._reward_one(i, action[i], action_raw[i])
            reward[i] = r
            failed[i] = fail
            done[i] = fail or (self.t[i] >= self.max_episode_length)
            for k in keys:
                log[k][i] = info[k]
            self.prev_action[i] = action[i]
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
            "final_observation": torch.from_numpy(
                np.nan_to_num(final_obs, nan=0.0, posinf=0.0, neginf=0.0)),
            "log": {
                "/reward_mean": float(reward.mean()),
                "/vx_mean": float(log["vx"].mean()),
                "/cmd_mean": float(log["cmd"].mean()),
                "/track_err": float(log["err"].mean()),
                "/x_mean": float(log["x"].mean()),
                "/x_abs_max": float(np.abs(log["x"]).max()),
                "/foot_z_mean": float(log["foot_z"].mean()),
                "/foot_y_mean": float(np.abs(log["foot_y"]).mean()),
                "/bend_max": float(log["bend"].max()),
                "/failed": float(failed.mean()),
                **{f"/{k}": float(log[k].mean()) for k in
                   ("r_vel", "r_vel_l2", "r_prog", "r_rail", "r_foot", "r_lat",
                    "r_effort", "r_rate", "r_ten", "r_action_l2")},
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
