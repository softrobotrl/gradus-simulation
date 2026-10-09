"""Stage 2: robot.xml -> batched MJX environment driven by servo (drum) angles.

Two classes over the same simulation:

TendonLegEnv: PyTorch, batched, auto-resetting; same interface as go2_env.Go2Env, so it plugs into rsl-rl:
    env = TendonLegEnv("pipeline/configs/opencr_tdcr_spatial.yaml", num_envs=64)
    obs = env.reset()                                   # TensorDict {"policy": (num_envs, num_obs)}
    obs, rewards, dones, extras = env.step(actions)     # actions: torch (num_envs, num_actions) in [-1, 1]

JAXTendonLegEnv: pure-functional JAX, one env; compose with jax.jit / jax.vmap yourself:
    env = JAXTendonLegEnv("pipeline/configs/opencr_tdcr_spatial.yaml")
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, action)                     # action in [-1, 1]^n_servos, 0 = servo angle 0

Task (example only): keep the chain upright. reward = tip height / straight-chain height - effort.
"""

import os
from pathlib import Path
from typing import NamedTuple

# JAX grabs 75% of GPU memory at import by default, which starves PyTorch when both share a GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from pipeline.config import RobotConfig
from pipeline.stl_to_xml import DEFAULT_OUT, stl_to_xml


class State(NamedTuple):
    data: mjx.Data
    target: jax.Array  # servo angle setpoints after rate limiting (rad)
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    t: jax.Array


class JAXTendonLegEnv:
    def __init__(self, config_path: str | Path, episode_s: float = 4.0, init_bend_deg: float = 3.0,
                 init_servo_angle: float = -0.3, effort_cost: float = 0.01, rebuild: bool = False):
        self.cfg = RobotConfig.load(config_path)
        xml_path = DEFAULT_OUT / self.cfg.name / "robot.xml"
        if rebuild or not xml_path.exists():
            stl_to_xml(config_path)
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mx = mjx.put_model(self.mj_model)

        m = self.mj_model
        self.n_act = m.nu
        self.ctrl_low, self.ctrl_high = jp.array(m.actuator_ctrlrange[:, 0]), jp.array(m.actuator_ctrlrange[:, 1])
        self.n_substeps = round(self.cfg.sim.control_dt / m.opt.timestep)
        self.max_step = self.cfg.servo.max_speed * self.cfg.sim.control_dt
        self.max_steps = round(episode_s / self.cfg.sim.control_dt)
        self.init_bend = jp.deg2rad(init_bend_deg)
        self.init_servo_angle = init_servo_angle
        self.effort_cost = effort_cost

        ball = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL]
        self.ball_qpos = jp.array([m.jnt_qposadr[j] for j in ball])[:, None] + jp.arange(4)
        self.cable_qpos = jp.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.tip = m.site("tip").id
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        self.base_z = float(data.xpos[m.body("disk0").id, 2])
        self.height = float(data.site_xpos[self.tip, 2]) - self.base_z
        self.obs_size = int(self._obs(mjx.put_data(m, data), jp.zeros(m.nu)).shape[0])

    def _obs(self, data: mjx.Data, target: jax.Array) -> jax.Array:
        tension = data.actuator_force / self.cfg.servo.drum_radius  # what motor current measures
        return jp.concatenate([data.qpos, data.qvel, target, tension, data.site_xpos[self.tip]])

    def reset(self, key: jax.Array) -> State:
        # Random small bend per joint (axis-angle -> quaternion), tendons start slack.
        k_axis, k_angle = jax.random.split(key)
        n_ball = self.ball_qpos.shape[0]
        axis = jax.random.normal(k_axis, (n_ball, 3))
        axis = axis.at[:, 2].set(0.0)
        axis = axis / jp.linalg.norm(axis, axis=1, keepdims=True)
        angle = jax.random.uniform(k_angle, (n_ball, 1), maxval=self.init_bend)
        quat = jp.concatenate([jp.cos(angle / 2), jp.sin(angle / 2) * axis], axis=1)

        target = jp.full(self.n_act, self.init_servo_angle)
        qpos = jp.array(self.mj_model.qpos0).at[self.ball_qpos].set(quat)
        qpos = qpos.at[self.cable_qpos].set(target * self.cfg.servo.drum_radius)
        data = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=qpos, ctrl=target))
        zero = jp.zeros(())
        return State(data, target, self._obs(data, target), zero, zero, jp.zeros((), jp.int32))

    def step(self, state: State, action: jax.Array) -> State:
        # action 0 = servo angle 0 (cable just taut); +1 = full wind, -1 = full release.
        action = jp.clip(action, -1, 1)
        command = jp.where(action > 0, action * self.ctrl_high, -action * self.ctrl_low)
        target = state.target + jp.clip(command - state.target, -self.max_step, self.max_step)
        data = state.data.replace(ctrl=target)
        data = jax.lax.fori_loop(0, self.n_substeps, lambda _, d: mjx.step(self.mx, d), data)

        upright = (data.site_xpos[self.tip, 2] - self.base_z) / self.height
        effort = jp.mean(jp.square(target / self.ctrl_high))
        reward = upright - self.effort_cost * effort
        unstable = jp.isnan(data.qpos).any()
        t = state.t + 1
        done = jp.logical_or(t >= self.max_steps, unstable).astype(jp.float32)
        reward = jp.where(unstable, 0.0, reward)
        return State(data, target, self._obs(data, target), reward, done, t)


class TendonLegEnv:
    """PyTorch, batched view of JAXTendonLegEnv with the rsl-rl VecEnv interface.

    All num_envs copies step in one compiled JAX call; finished episodes reset inside that call. Arrays cross
    between JAX and PyTorch through DLPack, without a copy when both sit on the same device.
    """

    def __init__(self, config_path: str | Path, num_envs: int, seed: int = 0, **env_kwargs):
        import torch
        from tensordict import TensorDict

        self._torch, self._TensorDict = torch, TensorDict
        self.jax_env = JAXTendonLegEnv(config_path, **env_kwargs)
        self.num_envs = num_envs
        self.num_actions = self.jax_env.n_act
        self.num_obs = self.jax_env.obs_size
        self.max_episode_length = self.jax_env.max_steps
        self.device = "cuda" if jax.default_backend() == "gpu" else "cpu"
        self.cfg = self.jax_env.cfg.model_dump(mode="json")
        self.extras: dict = {}
        self._key = jax.random.PRNGKey(seed)
        self._reset_all = jax.jit(jax.vmap(self.jax_env.reset))
        self._step_all = jax.jit(self._step_and_autoreset)
        self.reset()

    def _step_and_autoreset(self, state: State, action: jax.Array, key: jax.Array):
        stepped = jax.vmap(self.jax_env.step)(state, action)
        fresh = jax.vmap(self.jax_env.reset)(jax.random.split(key, self.num_envs))
        done = stepped.done > 0

        def pick(new: jax.Array, old: jax.Array) -> jax.Array:
            return jp.where(done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old)

        time_out = stepped.t >= self.jax_env.max_steps  # done by time limit, not by blowing up
        return jax.tree.map(pick, fresh, stepped), stepped.reward, done, time_out

    def _to_torch(self, x: jax.Array):
        return self._torch.from_dlpack(x).clone()  # clone: torch may write in place, JAX buffers must not change

    def _to_jax(self, x) -> jax.Array:
        return jax.dlpack.from_dlpack(x.detach().contiguous())

    @property
    def episode_length_buf(self):
        return self._to_torch(self.state.t)

    @episode_length_buf.setter
    def episode_length_buf(self, value) -> None:  # rsl-rl randomises start lengths through this
        self.state = self.state._replace(t=self._to_jax(value.to(self._torch.int32)))

    def get_observations(self):
        return self._TensorDict({"policy": self._to_torch(self.state.obs)}, batch_size=[self.num_envs])

    def reset(self):
        self._key, key = jax.random.split(self._key)
        self.state = self._reset_all(jax.random.split(key, self.num_envs))
        return self.get_observations()

    def step(self, actions):
        self._key, key = jax.random.split(self._key)
        self.state, reward, done, time_out = self._step_all(self.state, self._to_jax(actions.float()), key)
        self.extras = {"time_outs": self._to_torch(time_out).float(), "log": {"/reward_mean": float(reward.mean())}}
        return self.get_observations(), self._to_torch(reward), self._to_torch(done), self.extras
