"""Stage 2: robot.xml -> batched MJX environment driven by servo (drum) angles.

Two classes over the same simulation:

TendonLegEnv: PyTorch, batched, auto-resetting; same interface as go2_env.Go2Env, so it plugs into rsl-rl:
    env = TendonLegEnv("pipeline/configs/opencr_tdcr_spatial.yaml", num_envs=64)   # device="auto": GPU if usable
    obs = env.reset()                                   # TensorDict {"policy": (num_envs, num_obs)}
    obs, rewards, dones, extras = env.step(actions)     # actions: torch (num_envs, num_actions) in [-1, 1]
    # extras["time_outs"]: 1 where the episode ended on the time limit (not a failure)
    # extras["final_observation"]: the last obs of every env, before the auto-reset replaced it (for SAC/TD targets)

JAXTendonLegEnv: pure-functional JAX, one env; compose with jax.jit / jax.vmap yourself:
    env = JAXTendonLegEnv("pipeline/configs/opencr_tdcr_spatial.yaml")
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, action)                     # action in [-1, 1]^n_servos, 0 = servo angle 0

Task (example only): keep the chain upright. reward = tip height / straight-chain height - effort.

New tasks subclass JAXTendonLegEnv and override the hooks _load_model (add objects to the scene), _reset_scene
(their starting pose), _init_info (per-env task data at reset), _task (reward, failure, early end) and _obs;
see pipeline/board_env.py.
"""

import os
import warnings
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
    # Task data carried between steps. Always holds "terminated" (1 = episode ended by failure, not by the time
    # limit) and "metrics" (dict of per-env scalars that TendonLegEnv averages into extras["log"]).
    info: dict


class JAXTendonLegEnv:
    def __init__(self, config_path: str | Path, episode_s: float = 4.0, init_bend_deg: float = 3.0,
                 init_servo_angle: float = -0.3, effort_cost: float = 0.01, rebuild: bool = False,
                 action_filter: float = 1.0, device: jax.Device | None = None):
        """
        action_filter: each step the servo command moves this fraction of the way from the previous setpoint to
            the commanded angle (1 = instantly). Models servo bandwidth: 0.2 is a ~0.1 s time constant (~2 Hz),
            which stops policies from exploiting instant full-speed reversals (e.g. 25 Hz vibration).
        """
        self.cfg = RobotConfig.load(config_path)
        xml_path = DEFAULT_OUT / self.cfg.name / "robot.xml"
        if rebuild or not xml_path.exists():
            stl_to_xml(config_path)
        self.mj_model = self._load_model(xml_path)
        self.mx = mjx.put_model(self.mj_model, device=device)

        m = self.mj_model
        self.n_act = m.nu
        self.ctrl_low, self.ctrl_high = jp.array(m.actuator_ctrlrange[:, 0]), jp.array(m.actuator_ctrlrange[:, 1])
        self.dt = self.cfg.sim.control_dt
        self.n_substeps = round(self.dt / m.opt.timestep)
        self.max_step = self.cfg.servo.max_speed * self.dt
        self.max_steps = round(episode_s / self.dt)
        self.init_bend = jp.deg2rad(init_bend_deg)
        self.init_servo_angle = init_servo_angle
        self.effort_cost = effort_cost
        self.action_filter = action_filter

        ball = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL]
        self.ball_qpos = jp.array([m.jnt_qposadr[j] for j in ball])[:, None] + jp.arange(4)
        self.cable_qpos = jp.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.tip = m.site("tip").id
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        self.base_z = float(data.xpos[m.body("disk0").id, 2])
        self.height = float(data.site_xpos[self.tip, 2]) - self.base_z
        # Shape-only trace of reset: finds the obs size without running any physics.
        self.obs_size = int(jax.eval_shape(self.reset, jax.random.PRNGKey(0)).obs.shape[0])

    # ----- hooks a task subclass overrides -----

    def _load_model(self, xml_path: Path) -> mujoco.MjModel:
        """Compile the scene. Subclasses edit mujoco.MjSpec.from_file(xml_path) to add objects."""
        return mujoco.MjModel.from_xml_path(str(xml_path))

    def _reset_scene(self, key: jax.Array, qpos: jax.Array) -> jax.Array:
        """Set the starting qpos of anything the task added (the robot part is already set)."""
        return qpos

    def _init_info(self, key: jax.Array, data: mjx.Data) -> dict:
        """Per-env task data created at reset. Must have the same keys and shapes as _task returns."""
        return {"terminated": jp.zeros(()), "metrics": {"upright": jp.zeros(())}}

    def _task(self, state: State, data: mjx.Data, target: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict]:
        """(reward, failed, truncated, new info) after one control step. `state` is the state before the step.

        failed: the episode ends and its future is worth nothing (e.g. the robot broke): SAC does not bootstrap.
        truncated: the episode ends early for a reason outside the task (e.g. ran out of board), like a time-out:
        SAC still bootstraps, because the robot could have kept going.
        """
        upright = (data.site_xpos[self.tip, 2] - self.base_z) / self.height
        reward = upright - self.effort_cost * self._effort(target)
        return reward, jp.zeros(()), jp.zeros(()), {"terminated": jp.zeros(()), "metrics": {"upright": upright}}

    def _obs(self, data: mjx.Data, target: jax.Array, info: dict) -> jax.Array:
        tension = data.actuator_force / self.cfg.servo.drum_radius  # what motor current measures
        return jp.concatenate([data.qpos, data.qvel, target, tension, data.site_xpos[self.tip]])

    # ----- shared machinery -----

    def _effort(self, target: jax.Array) -> jax.Array:
        return jp.mean(jp.square(target / self.ctrl_high))

    def _reset_robot(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """qpos with a random small bend per joint (axis-angle -> quaternion), tendons slack; and servo targets."""
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
        return qpos, target

    def reset_qpos(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Starting qpos (robot + task objects) and servo targets. Pure JAX, no physics."""
        qpos, target = self._reset_robot(key)
        return self._reset_scene(jax.random.fold_in(key, 1), qpos), target

    def finish_reset(self, key: jax.Array, data, target: jax.Array) -> State:
        """Task info and obs for a freshly reset `data` (after forward kinematics)."""
        info = self._init_info(jax.random.fold_in(key, 2), data)
        zero = jp.zeros(())
        return State(data, target, self._obs(data, target, info), zero, zero, jp.zeros((), jp.int32), info)

    def reset(self, key: jax.Array) -> State:
        qpos, target = self.reset_qpos(key)
        data = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=qpos, ctrl=target))
        return self.finish_reset(key, data, target)

    def servo_targets(self, state: State, action: jax.Array) -> jax.Array:
        """Turn the action into rate-limited servo angle setpoints (rad)."""
        # action 0 = servo angle 0 (cable just taut); +1 = full wind, -1 = full release.
        action = jp.clip(action, -1, 1)
        command = jp.where(action > 0, action * self.ctrl_high, -action * self.ctrl_low)
        step = self.action_filter * (command - state.target)  # first-order servo response
        return state.target + jp.clip(step, -self.max_step, self.max_step)  # then the servo speed limit

    def step(self, state: State, action: jax.Array) -> State:
        target = self.servo_targets(state, action)
        data = state.data.replace(ctrl=target)
        data = jax.lax.fori_loop(0, self.n_substeps, lambda _, d: mjx.step(self.mx, d), data)
        return self.finish_step(state, data, target)

    def finish_step(self, state: State, data, target: jax.Array) -> State:
        """Reward, done and obs once the physics of a control step has run. Shared by the MJX and C backends."""
        reward, failed, truncated, info = self._task(state, data, target)
        unstable = jp.isnan(data.qpos).any()
        failed = jp.logical_or(failed > 0, unstable)
        info = {**info, "terminated": failed.astype(jp.float32)}
        t = state.t + 1
        done = (jp.logical_or(t >= self.max_steps, truncated > 0) | failed).astype(jp.float32)
        reward = jp.where(unstable, 0.0, reward)
        return State(data, target, self._obs(data, target, info), reward, done, t, info)


def gpu_problem() -> str | None:
    """None if both JAX and PyTorch can use a GPU; otherwise a short reason why not."""
    import torch

    try:
        gpus = jax.devices("gpu")
    except RuntimeError:  # this JAX install has no GPU backend (e.g. native Windows, or CUDA not working)
        gpus = []
    if not gpus:
        return "JAX found no usable GPU"
    if not torch.cuda.is_available():
        return "JAX sees a GPU but PyTorch does not (CPU-only torch build?)"
    return None


def pick_device(device: str = "auto", name: str = "env") -> tuple[jax.Device, str]:
    """Choose where the simulation runs. Returns (JAX device, matching torch device string).

    "auto": the GPU if both JAX and PyTorch can use it, otherwise the CPU. "gpu": GPU or raise. "cpu": CPU.
    JAX has no CUDA build for native Windows, so on Windows "auto" picks the CPU; run under WSL2 to use the GPU.
    """
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', got {device!r}")
    cpu = jax.devices("cpu")[0]
    if device == "cpu":
        print(f"{name}: running on CPU (device='cpu')")
        return cpu, "cpu"
    problem = gpu_problem()
    if problem is None:
        gpu = jax.devices("gpu")[0]
        print(f"{name}: running on GPU ({gpu.device_kind})")
        return gpu, f"cuda:{gpu.id}"
    if device == "gpu":
        raise RuntimeError(f"{name}: device='gpu' requested but {problem}. See pipeline/BOARD_TASK.md, "
                           "'Running on your GPU'.")
    warnings.warn(f"{name}: {problem}; running on CPU (slow). See pipeline/BOARD_TASK.md, 'Running on your GPU'.")
    print(f"{name}: running on CPU ({problem})")
    return cpu, "cpu"


class TendonLegEnv:
    """PyTorch, batched view of JAXTendonLegEnv with the rsl-rl VecEnv interface.

    All num_envs copies step in one compiled JAX call; finished episodes reset inside that call. Arrays cross
    between JAX and PyTorch through DLPack, without a copy when both sit on the same device.
    A task subclass only sets jax_env_cls (see board_env.BoardPushEnv).
    """

    jax_env_cls = JAXTendonLegEnv

    def __init__(self, config_path: str | Path, num_envs: int, seed: int = 0, device: str = "auto", **env_kwargs):
        import torch
        from tensordict import TensorDict

        self._torch, self._TensorDict = torch, TensorDict
        self._jax_device, self.device = pick_device(device, type(self).__name__)
        with jax.default_device(self._jax_device):
            self.jax_env = self.jax_env_cls(config_path, device=self._jax_device, **env_kwargs)
        self.num_envs = num_envs
        self.num_actions = self.jax_env.n_act
        self.num_obs = self.jax_env.obs_size
        self.max_episode_length = self.jax_env.max_steps
        self.cfg = self.jax_env.cfg.model_dump(mode="json")
        self.extras: dict = {}
        self._key = jax.device_put(jax.random.PRNGKey(seed), self._jax_device)
        self._reset_all = jax.jit(jax.vmap(self.jax_env.reset))
        self._step_all = jax.jit(self._step_and_autoreset)
        self.reset()

    def _step_and_autoreset(self, state: State, action: jax.Array, key: jax.Array):
        stepped = jax.vmap(self.jax_env.step)(state, action)
        fresh = jax.vmap(self.jax_env.reset)(jax.random.split(key, self.num_envs))
        done = stepped.done > 0

        def pick(new: jax.Array, old: jax.Array) -> jax.Array:
            return jp.where(done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old)

        # Done but not by failure = ended by the time limit: the value of the next state still counts.
        time_out = jp.logical_and(done, stepped.info["terminated"] == 0)
        final_obs = jp.nan_to_num(stepped.obs)  # a blown-up env is terminated anyway; keep NaN out of buffers
        metrics = jax.tree.map(jp.mean, stepped.info["metrics"])
        return jax.tree.map(pick, fresh, stepped), stepped.reward, done, time_out, final_obs, metrics

    def _to_torch(self, x: jax.Array):
        return self._torch.from_dlpack(x).clone()  # clone: torch may write in place, JAX buffers must not change

    def _to_jax(self, x) -> jax.Array:
        return jax.dlpack.from_dlpack(x.detach().to(self.device).contiguous())

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
        self.state, reward, done, time_out, final_obs, metrics = self._step_all(
            self.state, self._to_jax(actions.float()), key)
        log = {"/reward_mean": float(reward.mean())} | {f"/{k}": float(v) for k, v in metrics.items()}
        self.extras = {"time_outs": self._to_torch(time_out).float(),
                       "final_observation": self._to_torch(final_obs), "log": log}
        return self.get_observations(), self._to_torch(reward), self._to_torch(done), self.extras
