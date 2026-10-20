"""MJX (GPU) backend for the Gradus dog environment.

Same task, same observation layout and same action mapping as the C MuJoCo
backend in :mod:`pipeline.dog_env`, but the physics runs in MuJoCo MJX, so a few
thousand environments step inside one compiled call on the GPU.

Usage::

    from pipeline.dog_mjx_env import make_dog_mjx_env
    env = make_dog_mjx_env(1024, seed=0)
    obs = env.reset()["policy"]                       # (1024, 82)
    obs, reward, done, extras = env.step(actions)    # actions: (1024, 8) in [-1, 1]

The C backend stays the reference implementation; this one is checked against it
(same nominal stand, same tendon tensions, same observation layout).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import NamedTuple

# Let XLA allocate on demand: preallocating 75% of the GPU breaks when another
# process (or PyTorch's learner) already holds memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from pipeline.dog_env import FOOT_GEOMS, HINGE_NAMES, DogEnvConfig, obs_layout
from pipeline.dog_model import DEFAULT_OUT, DogModelConfig, build_dog_xml

_GRAVITY = jp.array([0.0, 0.0, -1.0])
# A foot counts as in contact when it is within this distance of the other geom.
_CONTACT_DIST = 1e-4


class DogState(NamedTuple):
    """Everything one environment carries between steps."""

    data: mjx.Data
    t: jax.Array
    phase: jax.Array
    command: jax.Array
    prev_action: jax.Array
    prev_target: jax.Array
    nominal: jax.Array
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    time_out: jax.Array
    terminated: jax.Array
    metrics: dict


def _euler_to_quat(roll, pitch, yaw):
    """MuJoCo [w, x, y, z] quaternion from ZYX Euler angles."""
    cr, sr = jp.cos(roll / 2), jp.sin(roll / 2)
    cp, sp = jp.cos(pitch / 2), jp.sin(pitch / 2)
    cy, sy = jp.cos(yaw / 2), jp.sin(yaw / 2)
    return jp.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class JAXDogEnv:
    """One dog written as pure JAX functions; compose with jit/vmap (see :class:`DogEnvMJX`)."""

    def __init__(self, cfg: DogEnvConfig, device: jax.Device | None = None):
        self.cfg = cfg
        model_path = Path(cfg.model_path) if cfg.model_path else DEFAULT_OUT / "robot.xml"
        if cfg.rebuild or not model_path.exists():
            build_dog_xml(model_path.parent, DogModelConfig())
        m = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_model = m
        self.mx = mjx.put_model(m, device=device)

        self.n_substeps = round(cfg.control_dt / m.opt.timestep)
        if self.n_substeps < 1:
            raise ValueError("control_dt must be a multiple of the model timestep")
        self.max_steps = max(1, round(cfg.episode_s / cfg.control_dt))
        self.max_step = cfg.max_speed * cfg.control_dt

        self.hinge_qpos = jp.array([m.jnt_qposadr[m.joint(n).id] for n in HINGE_NAMES])
        self.hinge_dof = jp.array([m.jnt_dofadr[m.joint(n).id] for n in HINGE_NAMES])
        self.cable_qpos = jp.array([m.jnt_qposadr[m.joint(m.actuator_trnid[a, 0]).id] for a in range(m.nu)])
        self.trunk_body = m.body("trunk").id
        self.root_dof = int(m.jnt_dofadr[m.joint("root").id])
        self.ctrl_low = jp.array(m.actuator_ctrlrange[:, 0])
        self.ctrl_high = jp.array(m.actuator_ctrlrange[:, 1])
        self.q0 = jp.zeros(len(HINGE_NAMES))
        self.tendon_rest = jp.array(m.tendon_range[:, 1])

        # Constant tendon Jacobian at the stand pose, per leg: rows = tendons, cols = joints.
        self.Jleg = jp.array(_tendon_jacobian(m, np.asarray(self.hinge_qpos)))

        # Open-loop trot phase offsets: FL and RR together, FR and RL opposite.
        leg_phase = np.array([0.0, math.pi, math.pi, 0.0])
        tendon_phase = np.array([0.0, math.pi / 2])
        self.gait_phase = jp.array(np.repeat(leg_phase, 2) + np.tile(tendon_phase, 4))

        # geom id -> foot index, -1 for everything else.
        geom_to_foot = -np.ones(m.ngeom, dtype=np.int64)
        for i, name in enumerate(FOOT_GEOMS):
            geom_to_foot[m.geom(name).id] = i
        self.geom_to_foot = jp.array(geom_to_foot)

        self.layout = obs_layout()
        self.num_obs = max(s.stop for s in self.layout.values())
        self.num_actions = m.nu

    # ----- controller -----

    def nominal_angles(self, q, qd, phase):
        """Nominal drum angles (rad) for one environment."""
        cfg = self.cfg
        if cfg.nominal_mode == "none":
            return jp.zeros(self.num_actions)
        base = jp.full((self.num_actions,), cfg.stand_angle)
        if cfg.gait_amp:
            base = base + cfg.gait_amp * jp.sin(phase + self.gait_phase)
        if cfg.nominal_mode == "pd":
            dq = cfg.kp_q * (self.q0 - q) - cfg.kd_q * qd
            d_l = jp.einsum("lij,lj->li", self.Jleg, dq.reshape(4, 2))
            base = base + d_l.reshape(self.num_actions) / cfg.drum_radius
        return jp.clip(base, self.ctrl_low, self.ctrl_high)

    # ----- state helpers -----

    def _gravity(self, data):
        return data.xmat[self.trunk_body].T @ _GRAVITY

    def _foot_contacts(self, data):
        contact = data._impl.contact
        dist, geom = contact.dist, contact.geom
        f1 = self.geom_to_foot[geom[:, 0]]
        f2 = self.geom_to_foot[geom[:, 1]]
        touching = (dist < _CONTACT_DIST) & ((f1 >= 0) | (f2 >= 0))
        idx = jp.where(f1 >= 0, f1, f2)
        onehot = (idx[:, None] == jp.arange(len(FOOT_GEOMS))[None, :]) & touching[:, None]
        return jp.max(onehot, axis=0).astype(jp.float32)

    def _obs(self, data, nominal, command, phase, action, target):
        cfg = self.cfg
        qpos, qvel = data.qpos, data.qvel
        R = data.xmat[self.trunk_body]
        gravity = R.T @ _GRAVITY
        v_world = qvel[self.root_dof:self.root_dof + 3]
        v_body = R.T @ v_world
        w_body = qvel[self.root_dof + 3:self.root_dof + 6]
        tension = data.actuator_force / cfg.drum_radius
        velocity_residual = jp.concatenate([v_body[:2] - command[:2], w_body[2:3] - command[2:3]])
        state_residual = jp.concatenate([
            (qpos[2] - cfg.height_target)[None],
            gravity - _GRAVITY,
            qpos[self.hinge_qpos] - self.q0,
            qvel[self.hinge_dof],
            velocity_residual,
            data.ten_length - self.tendon_rest,
        ])
        raw_state = jp.concatenate([
            qpos[3:7],
            v_body,
            w_body,
            tension / cfg.max_tension,
            self._foot_contacts(data),
        ])
        phase_sc = jp.stack([jp.sin(phase), jp.cos(phase)])
        action_context = jp.concatenate([
            nominal,
            target - nominal,
            action,
            phase_sc,
            command,
        ])
        return jp.concatenate([state_residual, raw_state, action_context]).astype(jp.float32)

    def _reward(self, data, target, action, prev_target, command):
        cfg = self.cfg
        qpos, qvel = data.qpos, data.qvel
        R = data.xmat[self.trunk_body]
        gravity = R.T @ _GRAVITY
        v_body = R.T @ qvel[self.root_dof:self.root_dof + 3]
        w_body = qvel[self.root_dof + 3:self.root_dof + 6]
        q = qpos[self.hinge_qpos]

        height = jp.exp(-jp.square((qpos[2] - cfg.height_target) / cfg.height_sigma))
        tilt = jp.linalg.norm(gravity[:2])
        upright = jp.exp(-jp.square(tilt / cfg.upright_sigma))
        vel_err = jp.square(v_body[0] - command[0]) + jp.square(v_body[1] - command[1])
        yaw_err = jp.square(w_body[2] - command[2])
        velocity = jp.exp(-vel_err / cfg.velocity_sigma**2)
        yaw = jp.exp(-yaw_err / cfg.yaw_sigma**2)
        posture = jp.exp(-jp.mean(jp.square(q)) / cfg.posture_sigma**2)
        tension_norm = (data.actuator_force / cfg.drum_radius) / cfg.max_tension
        tension_cost = jp.mean(jp.square(tension_norm))
        rate = jp.mean(jp.square((target - prev_target) / self.max_step))
        residual = jp.mean(jp.square(action))
        reward = (cfg.alive_reward + cfg.height_reward * height + cfg.upright_reward * upright
                  + cfg.velocity_reward * velocity + cfg.yaw_reward * yaw + cfg.posture_reward * posture
                  - cfg.tension_cost * tension_cost - cfg.rate_cost * rate - cfg.residual_cost * residual)
        reward = jp.where(jp.isfinite(reward), reward, 0.0)
        failed = ((qpos[2] < cfg.terminate_height) | (tilt > cfg.terminate_tilt)
                  | ~jp.isfinite(qpos).all())
        speed = jp.linalg.norm(v_body[:2])
        metrics = {
            "base_height": qpos[2],
            "tilt": tilt,
            "speed": speed,
            "vx": v_body[0],
            "vy": v_body[1],
            "wz": w_body[2],
            "height_reward": height,
            "upright_reward": upright,
            "velocity_reward": velocity,
            "yaw_reward": yaw,
            "posture_reward": posture,
            "tension": jp.mean(tension_norm),
            "action_rate": rate,
            "residual": jp.mean(jp.abs(action)),
            "foot_contact": jp.mean(self._foot_contacts(data)),
        }
        return reward, failed, metrics

    # ----- reset and step -----

    def reset(self, key):
        cfg = self.cfg
        k_h, k_rp, k_y, k_j, k_v, k_c, k_p = jax.random.split(key, 7)
        qpos = jp.array(self.mj_model.qpos0)
        qvel = jp.zeros(self.mj_model.nv)

        qpos = qpos.at[2].set(qpos[2] + jax.random.uniform(
            k_h, (), minval=-cfg.init_height_noise, maxval=cfg.init_height_noise))
        roll = jax.random.uniform(k_rp, (), minval=-cfg.init_tilt_noise, maxval=cfg.init_tilt_noise)
        pitch = jax.random.uniform(k_rp, (), minval=-cfg.init_tilt_noise, maxval=cfg.init_tilt_noise)
        yaw = (jax.random.uniform(k_y, (), minval=-cfg.init_yaw_noise, maxval=cfg.init_yaw_noise)
               if cfg.init_yaw_noise else 0.0)
        qpos = qpos.at[3:7].set(_euler_to_quat(roll, pitch, yaw))

        joint_noise = jax.random.uniform(k_j, (len(HINGE_NAMES),),
                                         minval=-cfg.init_joint_noise, maxval=cfg.init_joint_noise)
        qpos = qpos.at[self.hinge_qpos].set(self.q0 + joint_noise)
        qvel = qvel.at[self.root_dof:self.root_dof + 6].set(jax.random.uniform(
            k_v, (6,), minval=-cfg.init_vel_noise, maxval=cfg.init_vel_noise))
        qvel = qvel.at[self.hinge_dof].set(jax.random.uniform(
            k_v, (len(HINGE_NAMES),), minval=-cfg.init_vel_noise, maxval=cfg.init_vel_noise))

        if cfg.command_mode == "velocity":
            command = jp.array([
                jax.random.uniform(k_c, (), minval=cfg.vx_range[0], maxval=cfg.vx_range[1]),
                jax.random.uniform(k_c, (), minval=cfg.vy_range[0], maxval=cfg.vy_range[1]),
                jax.random.uniform(k_c, (), minval=cfg.wz_range[0], maxval=cfg.wz_range[1]),
            ])
        else:
            command = jp.zeros(3)
        phase = (jax.random.uniform(k_p, (), minval=0.0, maxval=2 * math.pi) if cfg.gait_hz else 0.0)

        nominal = self.nominal_angles(qpos[self.hinge_qpos], qvel[self.hinge_dof], phase)
        qpos = qpos.at[self.cable_qpos].set(cfg.drum_radius * nominal)
        data = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=qpos, qvel=qvel, ctrl=nominal))

        zero = jp.zeros(())
        obs = self._obs(data, nominal, command, phase, jp.zeros(self.num_actions), nominal)
        metrics = {k: zero for k in ("base_height", "tilt", "speed", "vx", "vy", "wz", "height_reward",
                                     "upright_reward", "velocity_reward", "yaw_reward", "posture_reward",
                                     "tension", "action_rate", "residual", "foot_contact")}
        return DogState(data, jp.zeros((), jp.int32), phase, command, jp.zeros(self.num_actions), nominal,
                        nominal, obs, zero, zero, zero, zero, metrics)

    def step(self, state, action):
        cfg = self.cfg
        action = jp.clip(action, -1.0, 1.0)
        data = state.data
        q = data.qpos[self.hinge_qpos]
        qd = data.qvel[self.hinge_dof]
        phase = state.phase
        if cfg.gait_hz:
            phase = jp.mod(phase + 2 * math.pi * cfg.gait_hz * cfg.control_dt, 2 * math.pi)

        nominal = self.nominal_angles(q, qd, phase)
        command = jp.clip(nominal + cfg.residual_scale * action, self.ctrl_low, self.ctrl_high)
        filtered = state.prev_target + cfg.action_filter * (command - state.prev_target)
        target = state.prev_target + jp.clip(filtered - state.prev_target, -self.max_step, self.max_step)
        target = jp.clip(target, self.ctrl_low, self.ctrl_high)

        data = data.replace(ctrl=target)
        data = jax.lax.fori_loop(0, self.n_substeps, lambda _, d: mjx.step(self.mx, d), data)

        reward, failed, metrics = self._reward(data, target, action, state.prev_target, state.command)
        t = state.t + 1
        done = failed | (t >= self.max_steps)
        time_out = done & ~failed
        obs = self._obs(data, nominal, state.command, phase, action, target)
        return DogState(data, t, phase, state.command, action, target, nominal, obs, reward, done,
                        time_out, failed, metrics)


def _tendon_jacobian(m: mujoco.MjModel, hinge_qpos: np.ndarray) -> np.ndarray:
    """d(tendon length)/d(joint angle) at qpos0, as a (4 legs, 2 tendons, 2 joints) array."""
    data = mujoco.MjData(m)
    eps = 1e-6
    J = np.zeros((m.ntendon, len(hinge_qpos)), dtype=np.float64)
    for j, adr in enumerate(hinge_qpos):
        qp = m.qpos0.copy()
        qp[adr] += eps
        data.qpos[:] = qp
        mujoco.mj_forward(m, data)
        lp = data.ten_length.copy()
        qm = m.qpos0.copy()
        qm[adr] -= eps
        data.qpos[:] = qm
        mujoco.mj_forward(m, data)
        lm = data.ten_length.copy()
        J[:, j] = (lp - lm) / (2 * eps)
    # Rows are tendon index (FL_t0, FL_t1, FR_t0, ...); columns are joint index
    # (FL_hip, FL_knee, FR_hip, ...).  The matrix is block diagonal per leg.
    return np.array([J[2 * leg:2 * leg + 2, 2 * leg:2 * leg + 2] for leg in range(4)])


class DogEnvMJX:
    """Batched MJX dog environment with the same interface as :class:`pipeline.dog_env.DogEnv`."""

    def __init__(self, num_envs: int, config: DogEnvConfig | None = None, seed: int = 0,
                 device: str = "auto", **kwargs):
        import torch
        from tensordict import TensorDict

        if config is None:
            config = DogEnvConfig(**kwargs)
        elif kwargs:
            raise ValueError("pass either config=... or keyword fields, not both")
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self._torch, self._TensorDict = torch, TensorDict
        self.cfg = config
        self.num_envs = int(num_envs)

        if device == "auto":
            device = "gpu" if jax.devices("gpu") else "cpu"
        if device == "gpu":
            self._jax_device = jax.devices("gpu")[0]
            self.device = f"cuda:{self._jax_device.id}"
        elif device == "cpu":
            self._jax_device = jax.devices("cpu")[0]
            self.device = "cpu"
        else:
            raise ValueError("device must be 'auto', 'gpu' or 'cpu'")

        with jax.default_device(self._jax_device):
            self.jax_env = JAXDogEnv(config, device=self._jax_device)
        self.num_obs = self.jax_env.num_obs
        self.num_actions = self.jax_env.num_actions
        self.max_episode_length = self.jax_env.max_steps
        self.dt = config.control_dt
        self.extras: dict = {}

        self._key = jax.device_put(jax.random.PRNGKey(seed), self._jax_device)
        self._reset_all = jax.jit(jax.vmap(self.jax_env.reset))
        self._step_all = jax.jit(self._step_and_autoreset)
        self.state = None
        self.reset()

    # ----- torch <-> jax -----

    def _to_torch(self, x):
        return self._torch.from_dlpack(x).clone()

    def _to_jax(self, x):
        if not isinstance(x, self._torch.Tensor):
            x = self._torch.as_tensor(x, dtype=self._torch.float32)
        return jax.dlpack.from_dlpack(x.detach().to(self.device).contiguous())

    # ----- vec env interface -----

    @property
    def obs_layout(self) -> dict[str, slice]:
        return dict(self.jax_env.layout)

    @property
    def episode_length_buf(self):
        return self._to_torch(self.state.t)

    @episode_length_buf.setter
    def episode_length_buf(self, value) -> None:
        self.state = self.state._replace(t=self._to_jax(value.to(self._torch.int32)))

    def _step_and_autoreset(self, state, action, key):
        stepped = jax.vmap(self.jax_env.step)(state, action)
        fresh = jax.vmap(self.jax_env.reset)(jax.random.split(key, self.num_envs))
        done = stepped.done > 0

        def pick(new, old):
            return jp.where(done.reshape(done.shape + (1,) * (jp.ndim(old) - 1)), new, old)

        final_obs = jp.nan_to_num(stepped.obs)
        metrics = jax.tree.map(jp.mean, stepped.metrics)
        return jax.tree.map(pick, fresh, stepped), stepped.reward, stepped.done, stepped.time_out, final_obs, metrics

    def _obs_tensor(self):
        return self._TensorDict({"policy": self._to_torch(self.state.obs)}, batch_size=[self.num_envs])

    def reset(self):
        self._key, key = jax.random.split(self._key)
        self.state = self._reset_all(jax.random.split(key, self.num_envs))
        return self._obs_tensor()

    def step(self, actions):
        self._key, key = jax.random.split(self._key)
        self.state, reward, done, time_out, final_obs, metrics = self._step_all(
            self.state, self._to_jax(actions), key)
        log = {"/reward_mean": float(reward.mean())} | {f"/{k}": float(v) for k, v in metrics.items()}
        self.extras = {"time_outs": self._to_torch(time_out).float(),
                       "final_observation": self._to_torch(final_obs), "log": log}
        return self._obs_tensor(), self._to_torch(reward), self._to_torch(done), self.extras

    def close(self) -> None:
        pass


def make_dog_mjx_env(num_envs: int, seed: int = 0, device: str = "auto", **kwargs) -> DogEnvMJX:
    env = DogEnvMJX(num_envs, seed=seed, device=device, **kwargs)
    print(f"DogEnvMJX: MJX on {env.device}, {env.num_envs} envs")
    return env


__all__ = ["DogEnvMJX", "JAXDogEnv", "DogState", "make_dog_mjx_env"]
