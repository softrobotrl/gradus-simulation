"""C MuJoCo backend: the same tasks and PyTorch interface as env.TendonLegEnv, with the physics on CPU threads.

Why: MJX (JAX) is built for GPUs. On a CPU, one small robot steps far faster in regular C MuJoCo, and many
envs can step in parallel threads (MuJoCo releases Python's GIL while it simulates). Measured on a 20-core laptop
with random actions: ~14,000 transitions/s here vs ~85/s for MJX on the same CPU.

Only the physics moves. The task code (servo rate limit, reward, failure, observations) is the same JAX code
the MJX env runs, called through JAXTendonLegEnv.servo_targets / finish_step / reset_qpos / finish_reset, so
both backends compute identical rewards from identical states. The one rule for task code: read only the
MjData fields listed in SimData.

    from pipeline.board_env import BoardPushEnvCPU
    env = BoardPushEnvCPU("pipeline/configs/opencr_tdcr_spatial.yaml", num_envs=256)
    obs = env.reset()["policy"]
    obs, reward, done, extras = env.step(actions)       # same as BoardPushEnv
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from pipeline.env import JAXTendonLegEnv, State


class SimData(NamedTuple):
    """The MjData fields task code may read (same names as mjx.Data, so task code can't tell the difference)."""

    qpos: np.ndarray
    qvel: np.ndarray
    actuator_force: np.ndarray
    site_xpos: np.ndarray
    geom_xpos: np.ndarray


# MuJoCo resets an env by itself when the solver diverges and records it in these warning counters.
_DIVERGED = [mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL]


class CpuTendonLegEnv:
    """Batched, auto-resetting env with the TendonLegEnv (rsl-rl VecEnv) interface, physics in C MuJoCo threads.

    Tensors are always on the CPU (env.device == "cpu"); move them to the learner's device yourself.
    A task subclass only sets jax_env_cls (see board_env.BoardPushEnvCPU).
    """

    jax_env_cls = JAXTendonLegEnv

    def __init__(self, config_path: str | Path, num_envs: int, seed: int = 0, num_threads: int | None = None,
                 **env_kwargs):
        """num_threads: physics threads (default: one per CPU core, at most num_envs). env_kwargs go to jax_env_cls."""
        import torch
        from tensordict import TensorDict

        self._torch, self._TensorDict = torch, TensorDict
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            self.jax_env = self.jax_env_cls(config_path, device=cpu, **env_kwargs)
        env, m = self.jax_env, self.jax_env.mj_model
        self.mj_model = m
        self.device = "cpu"
        self.num_envs = num_envs
        self.num_actions = env.n_act
        self.num_obs = env.obs_size
        self.max_episode_length = env.max_steps
        self.cfg = env.cfg.model_dump(mode="json")
        self.extras: dict = {}
        self._cpu = cpu
        self._key = jax.device_put(jax.random.PRNGKey(seed), cpu)

        self.datas = [mujoco.MjData(m) for _ in range(num_envs)]
        threads = max(1, min(num_threads or os.cpu_count() or 1, num_envs))
        self._chunks = [range(i, num_envs, threads) for i in range(threads)]
        self._pool = ThreadPoolExecutor(threads)
        self.num_threads = threads
        # Preallocated SimData buffers that the threads fill in place.
        self._buf = SimData(np.zeros((num_envs, m.nq), np.float32), np.zeros((num_envs, m.nv), np.float32),
                            np.zeros((num_envs, m.nu), np.float32), np.zeros((num_envs, m.nsite, 3), np.float32),
                            np.zeros((num_envs, m.ngeom, 3), np.float32))

        def stub(target, t=None, info=None):  # a State with only the fields task code reads
            return State(None, target, None, None, None, t, info)

        self._targets = jax.jit(jax.vmap(lambda target, a: env.servo_targets(stub(target), a)))
        self._finish_step = jax.jit(jax.vmap(
            lambda target, t, info, sim, new_target: env.finish_step(stub(target, t, info), sim, new_target)))
        self._reset_qpos = jax.jit(jax.vmap(env.reset_qpos))
        self._finish_reset = jax.jit(jax.vmap(env.finish_reset))
        self._merge = jax.jit(lambda done, new, old: jax.tree.map(
            lambda a, b: jp.where(done.reshape(done.shape + (1,) * (b.ndim - 1)), a, b), new, old))
        self._post = jax.jit(lambda s: (jp.logical_and(s.done > 0, s.info["terminated"] == 0), jp.nan_to_num(s.obs),
                                        jax.tree.map(jp.mean, s.info["metrics"])))
        self.reset()

    # ----- physics in threads -----

    def _run(self, fn) -> None:
        for future in [self._pool.submit(fn, chunk) for chunk in self._chunks]:
            future.result()  # re-raises any error from a thread

    def _gather_into(self, i: int, d: mujoco.MjData) -> None:
        b = self._buf
        b.qpos[i] = d.qpos
        if sum(d.warning[w].number for w in _DIVERGED):
            b.qpos[i] = np.nan  # the task code's NaN check marks this env as failed; it gets reset next
        b.qvel[i], b.actuator_force[i] = d.qvel, d.actuator_force
        b.site_xpos[i], b.geom_xpos[i] = d.site_xpos, d.geom_xpos

    def _sim(self) -> SimData:
        return SimData(*(jp.array(x) for x in self._buf))  # copy: the threads overwrite the buffers next step

    def _physics_step(self, target: np.ndarray) -> None:
        m, n_sub = self.mj_model, self.jax_env.n_substeps

        def work(chunk):
            for i in chunk:
                d = self.datas[i]
                d.ctrl[:] = target[i]
                mujoco.mj_step(m, d, nstep=n_sub)
                self._gather_into(i, d)

        self._run(work)

    def _physics_reset(self, which: np.ndarray, qpos: np.ndarray, target: np.ndarray) -> None:
        m = self.mj_model
        which = set(np.flatnonzero(which).tolist())

        def work(chunk):
            for i in chunk:
                if i in which:
                    d = self.datas[i]
                    mujoco.mj_resetData(m, d)
                    d.qpos[:], d.ctrl[:] = qpos[i], target[i]
                    mujoco.mj_forward(m, d)
                    self._gather_into(i, d)

        self._run(work)

    def _fresh(self, which: np.ndarray) -> State:
        """Reset the envs where `which` is True and return a fresh State for every env (use only those rows)."""
        self._key, key = jax.random.split(self._key)
        keys = jax.random.split(key, self.num_envs)
        qpos, target = self._reset_qpos(keys)
        self._physics_reset(which, np.asarray(qpos), np.asarray(target))
        return self._finish_reset(keys, self._sim(), target)

    # ----- TendonLegEnv interface -----

    def _to_torch(self, x):
        return self._torch.from_numpy(np.array(x))

    @property
    def episode_length_buf(self):
        return self._to_torch(self.state.t)

    @episode_length_buf.setter
    def episode_length_buf(self, value) -> None:
        self.state = self.state._replace(t=jp.asarray(value.cpu().numpy().astype(np.int32)))

    def get_observations(self):
        return self._TensorDict({"policy": self._to_torch(self.state.obs)}, batch_size=[self.num_envs])

    def reset(self):
        self.state = self._fresh(np.ones(self.num_envs, bool))
        return self.get_observations()

    def step(self, actions):
        actions = jp.asarray(actions.detach().float().cpu().numpy())
        target = self._targets(self.state.target, actions)
        self._physics_step(np.asarray(target))
        stepped = self._finish_step(self.state.target, self.state.t, self.state.info, self._sim(), target)
        done = np.asarray(stepped.done) > 0
        self.state = self._merge(jp.asarray(done), self._fresh(done), stepped) if done.any() else stepped
        time_out, final_obs, metrics = self._post(stepped)
        log = {"/reward_mean": float(stepped.reward.mean())} | {f"/{k}": float(v) for k, v in metrics.items()}
        self.extras = {"time_outs": self._to_torch(time_out).float(), "final_observation": self._to_torch(final_obs),
                       "log": log}
        return self.get_observations(), self._to_torch(stepped.reward), self._to_torch(done), self.extras
