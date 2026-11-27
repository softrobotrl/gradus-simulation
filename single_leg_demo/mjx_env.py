"""MJX (JAX) version of the single-leg rig env, for fast PPO training on a GPU.

The CPU env (rig_env.SingleLegRigEnv) stays the reference: this file copies
its physics setup, servo model, observations and reward (mjx_rewards.py), and
`tests/test_mjx_env.py` checks the two agree.  Train here, then check a policy
on the CPU env (reward_check/passes.py).

Episode starts come from a bank of reset states made by the CPU env itself
(``build_reset_bank``): its reset pose, settling and domain randomization run
unchanged there, and each MJX episode starts from a random bank entry.  So
randomized physics is redrawn every episode, as on the CPU env, from a finite
(default 4096) set of draws.

Usage (Brax PPO): see single_leg_demo/train_mjx.py.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from single_leg_demo import mjx_rewards, rewards
from single_leg_demo.rig_env import CMD_BUF, GRAVITY, N_TENDON, RigConfig, SingleLegRigEnv

# model fields that domain randomization changes per episode (rig_env.nom)
MODEL_FIELDS = ("geom_friction", "body_mass", "body_inertia", "body_pos", "dof_damping",
                "dof_frictionloss", "jnt_stiffness", "actuator_gainprm", "actuator_biasprm",
                "actuator_forcerange", "actuator_ctrlrange")
# per-episode servo / sensor draws copied from the CPU env
DRAW_FIELDS = ("strength", "calib", "battery", "sag", "delay", "acc_bias", "gyro_bias",
               "load_scale", "volt", "pull_bias")


def build_reset_bank(cfg: RigConfig, n: int, seed: int = 0, batch: int = 256) -> dict:
    """``n`` episode starts from the CPU env: physics state, randomized model
    fields and servo/sensor draws."""
    cfg = dataclasses.replace(cfg, num_envs=min(batch, n), seed=seed)
    env = SingleLegRigEnv(cfg)
    out: dict[str, list] = {k: [] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart",
                                            *MODEL_FIELDS, *DRAW_FIELDS)}
    try:
        while len(out["qpos"]) < n:
            env.reset()
            for i in range(env.num_envs):
                d, mi = env.datas[i], env.models[i]
                for k in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                    out[k].append(getattr(d, k).copy())
                for k in MODEL_FIELDS:
                    out[k].append(getattr(mi, k).copy())
                for k in DRAW_FIELDS:
                    out[k].append(np.array(getattr(env, k)[i], dtype=np.float64))
    finally:
        env.close()
    return {k: np.stack(v[:n]) for k, v in out.items()}


@dataclasses.dataclass(frozen=True)
class Consts:
    """Everything the JAX step needs from the CPU env (read once)."""
    l0: np.ndarray
    stroke: np.ndarray
    fmax: np.ndarray
    ctrl_lo: np.ndarray
    servo_of: np.ndarray
    cable_pairs: np.ndarray | None
    first_cable: np.ndarray
    servo_counts: np.ndarray | None
    fmax_nom: np.ndarray
    nom: dict
    n_servo: int
    slide_qadr: int
    ball_qadr: np.ndarray
    ball_dadr: np.ndarray
    tip_id: int
    slider_body: int
    imu_acc_adr: int
    imu_gyro_adr: int
    foot_force_adr: int
    foot_linvel_adr: int
    foot_angvel_adr: int
    foot_radius: float
    foot_geom: int
    slide_dof: int
    ball_jnt0: int
    ball_dof0: int
    timestep: float

    @staticmethod
    def from_env(env: SingleLegRigEnv) -> "Consts":
        return Consts(
            l0=env.l0, stroke=env.stroke, fmax=env.fmax, ctrl_lo=env.ctrl_lo,
            servo_of=env.servo_of, cable_pairs=env.cable_pairs, first_cable=env.first_cable,
            servo_counts=env.servo_counts, fmax_nom=env.nom["actuator_forcerange"][:, 0].copy(),
            nom={k: v.copy() for k, v in env.nom.items()}, n_servo=env.n_servo,
            slide_qadr=env.slide_qadr, ball_qadr=env.ball_qadr, ball_dadr=env.ball_dadr,
            tip_id=env.tip_id, slider_body=env.slider_body, imu_acc_adr=env.imu_acc_adr,
            imu_gyro_adr=env.imu_gyro_adr, foot_force_adr=env.foot_force_adr,
            foot_linvel_adr=env.foot_linvel_adr, foot_angvel_adr=env.foot_angvel_adr,
            foot_radius=env.foot_radius, foot_geom=env.foot_geom, slide_dof=env.slide_dof,
            ball_jnt0=int(env.ball_jnt[0]), ball_dof0=int(env.ball_dofs[0]),
            timestep=float(env.m.opt.timestep))


def _quat_to_rotvec(q):
    w = jnp.abs(q[..., 0])
    v = q[..., 1:]
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return 2.0 * jnp.arctan2(n, w[..., None]) * v / jnp.maximum(n, 1e-9)


class MjxRigEnv(Env):
    """One env (Brax vmaps it).  Observations: {'state': actor input,
    'privileged_state': critic input}, the same values as the CPU env."""

    def __init__(self, cfg: RigConfig, bank: dict | None = None, bank_size: int = 4096):
        if cfg.obs_mode != "hardware":
            raise ValueError("mjx env supports obs_mode='hardware' only")
        if not cfg.fixed_pull_bias:
            raise ValueError("mjx env needs fixed_pull_bias=True")
        self.cfg = cfg
        self.c_ = dict(dataclasses.asdict(cfg))
        ref = SingleLegRigEnv(dataclasses.replace(cfg, num_envs=1, num_threads=1))
        self.k = Consts.from_env(ref)
        self.m = ref.m
        self.num_obs, self.num_critic_obs = ref.num_obs, ref.num_critic_obs
        self.hw_frame = ref.hw_frame
        ref.close()
        self.mx = mjx.put_model(self.m)
        self.bank = bank if bank is not None else build_reset_bank(cfg, bank_size, seed=cfg.seed)
        self.bank_j = jax.tree.map(jnp.asarray, self.bank)
        self.n_bank = len(self.bank["qpos"])
        self.per_env_models = bool(cfg.dr_enabled or cfg.dr_override)
        self.data0 = mjx.make_data(self.mx)

    # ------------------------------------------------------------------ #
    @property
    def observation_size(self):
        return {"state": (self.num_obs,), "privileged_state": (self.num_critic_obs,)}

    @property
    def action_size(self) -> int:
        return self.k.n_servo

    @property
    def backend(self) -> str:
        return "mjx"

    # ------------------------------------------------------------------ #
    def _model(self, mf: dict):
        return self.mx.replace(**mf)

    def _set_cmd(self, s: dict, key) -> dict:
        c, dt = self.c_, self.cfg.control_dt
        period = jnp.maximum(s["cmd_period"], 1e-3)
        segment = jnp.floor(s["t"] * dt / period + 1e-9).astype(jnp.int32)
        new_mag = jax.random.uniform(key, (), minval=c["cmd_min"], maxval=c["cmd_max"])
        mag = jnp.where(c["cmd_resample"] & (segment != s["cmd_segment"]), new_mag, s["cmd_mag"])
        sign = s["cmd_sign0"] * jnp.where(segment % 2 == 0, 1.0, -1.0)
        cmd_raw = sign * mag
        if c["cmd_ramp"] > 0.0:
            step = c["cmd_ramp"] * dt
            cmd = s["cmd"] + jnp.clip(cmd_raw - s["cmd"], -step, step)
        else:
            cmd = cmd_raw
        return {**s, "cmd_mag": mag, "cmd_segment": segment, "cmd_raw": cmd_raw, "cmd": cmd}

    def _phase(self, s: dict):
        return jnp.mod(s["t"] * self.cfg.control_dt / jnp.maximum(s["cmd_period"], 1e-3), 1.0)

    def _hw_frame(self, d, s: dict, key) -> jnp.ndarray:
        k, c = self.k, self.c_
        target = jnp.clip((k.l0 - s["servo_cmd"]) / k.stroke, 0.0, 1.0)
        setpoint = (k.l0 - d.ctrl) / k.stroke
        taut = (k.l0 - d.ten_length) / k.stroke
        position = jnp.minimum(taut, setpoint) + s["calib"] / k.stroke
        load = -d.actuator_force / k.fmax
        if k.cable_pairs is not None:
            a, b = k.cable_pairs[:, 0], k.cable_pairs[:, 1]
            target = (target[a] + target[b]) / 2
            position = jnp.minimum(position[a], position[b])
            load = (load[a] + load[b]) / 2
        acc = d.sensordata[k.imu_acc_adr:k.imu_acc_adr + 3] / GRAVITY
        gyro = d.sensordata[k.imu_gyro_adr:k.imu_gyro_adr + 3]
        if c["dr_enabled"]:
            sc, S = c["dr_scale"], k.n_servo
            z = jax.random.normal(key, (6 + 2 * S,))
            acc = acc + s["acc_bias"] + c["dr_acc_noise"] * sc * z[0:3]
            gyro = gyro + s["gyro_bias"] + c["dr_gyro_noise"] * sc * z[3:6]
            position = position + c["dr_pos_noise"] * sc * z[6:6 + S]
            load = load * s["load_scale"][k.first_cable] + c["dr_load_noise"] * sc * z[6 + S:]
        if k.servo_counts is not None:
            position = jnp.round(position * k.servo_counts) / k.servo_counts
        position = jnp.clip(position, 0.0, 1.0)
        load = jnp.clip(load, 0.0, 1.5)
        acc = jnp.clip(acc, -c["acc_clip"], c["acc_clip"])
        gyro = jnp.clip(gyro / c["gyro_scale"], -2.0, 2.0)
        phase = self._phase(s)
        head = jnp.stack([s["cmd"] / max(c["obs_cmd_scale"], 1e-3),
                          jnp.sin(2 * jnp.pi * phase), jnp.cos(2 * jnp.pi * phase)])
        parts = [head, s["prev_action"], target, position]
        if c["servo_load_obs"]:
            parts.append(load)
        if c.get("imu_obs", True):          # no IMU on the confirmed rig (2026-11-26)
            parts += [acc, gyro]
        return jnp.concatenate(parts)

    def _critic(self, d, s: dict, mf: dict) -> jnp.ndarray:
        k, c = self.k, self.c_
        quat = d.qpos[k.ball_qadr[:, None] + np.arange(4)]
        rotvec = _quat_to_rotvec(quat).reshape(-1)
        qvel_ball = d.qvel[k.ball_dadr[:, None] + np.arange(3)].reshape(-1)
        phase = self._phase(s)
        parts = [
            jnp.stack([d.qpos[k.slide_qadr] / c["rail_half"], d.qvel[0] / 0.20,
                       s["cmd"] / max(c["obs_cmd_scale"], 1e-3),
                       jnp.sin(2 * jnp.pi * phase), jnp.cos(2 * jnp.pi * phase)]),
            rotvec, qvel_ball * 0.05,
            jnp.clip((k.l0 - d.ten_length) / k.stroke, -0.5, 1.5),
            jnp.clip(-d.actuator_force / k.fmax, 0.0, 1.5),
            jnp.clip((k.l0 - d.ctrl) / k.stroke, 0.0, 1.0),
            s["prev_action"],
        ]
        if c["critic_scored_speed"]:
            parts.append(jnp.reshape(s["vx_avg"] / max(c["obs_cmd_scale"], 1e-3), (1,)))
        if c["critic_dr_params"]:
            nom = k.nom
            parts.append(jnp.stack([
                mf["geom_friction"][k.foot_geom, 0],
                mf["dof_frictionloss"][k.slide_dof] / nom["dof_frictionloss"][k.slide_dof],
                mf["dof_damping"][k.slide_dof] / nom["dof_damping"][k.slide_dof],
                1000 * (mf["body_pos"][k.slider_body, 2] - nom["body_pos"][k.slider_body, 2]) / 10.0,
                mf["body_mass"][k.slider_body] / nom["body_mass"][k.slider_body],
                mf["jnt_stiffness"][k.ball_jnt0] / nom["jnt_stiffness"][k.ball_jnt0],
                mf["dof_damping"][k.ball_dof0] / nom["dof_damping"][k.ball_dof0],
                mf["actuator_gainprm"][0, 0] / nom["actuator_gainprm"][0, 0],
                jnp.mean(s["strength"]), s["battery"], s["sag"],
                s["delay"] * k.timestep * 1000 / 20.0, s["volt"],
            ]))
        return jnp.concatenate(parts)

    # ------------------------------------------------------------------ #
    def reset(self, rng: jax.Array) -> State:
        k, c = self.k, self.c_
        kb, km, ks, kp, kc, kn, krest = jax.random.split(rng, 7)
        b = jax.random.randint(kb, (), 0, self.n_bank)
        e = jax.tree.map(lambda x: x[b], self.bank_j)
        mf = {f: e[f] for f in MODEL_FIELDS}
        d = self.data0.replace(qpos=e["qpos"], qvel=e["qvel"], ctrl=e["ctrl"],
                               qacc_warmstart=e["qacc_warmstart"])
        d = mjx.forward(self._model(mf), d)
        jitter = c["cmd_period_jitter"]
        hold = k.l0 - k.stroke * e["pull_bias"]
        s = {
            "key": krest, "t": jnp.int32(0), "total_steps": jnp.int32(0),
            "cmd": jnp.zeros((), d.qpos.dtype), "cmd_raw": jnp.zeros((), d.qpos.dtype),
            "cmd_mag": jax.random.uniform(km, (), minval=c["cmd_min"], maxval=c["cmd_max"]),
            "cmd_sign0": jnp.where(jax.random.uniform(ks) < 0.5, 1.0, -1.0),
            "cmd_period": c["cmd_period"] * (1.0 + jax.random.uniform(kp, (), minval=-jitter, maxval=jitter)),
            "cmd_segment": jnp.int32(0),
            "servo_cmd": hold, "cmd_buf": jnp.tile(hold, (CMD_BUF, 1)),
            "prev_action": jnp.zeros(k.n_servo), "prev_action2": jnp.zeros(k.n_servo),
            "prev_tension": -d.actuator_force, "vx_avg": d.qvel[0], "vx_idle": d.qvel[0],
            "push_force": jnp.zeros((), d.qpos.dtype), "push_left": jnp.int32(0),
            "mf": mf,
            **{f: e[f] for f in DRAW_FIELDS},
        }
        s["delay"] = s["delay"].astype(jnp.int32)
        s = self._set_cmd(s, kc)
        frame = self._hw_frame(d, s, kn)
        s["hist"] = jnp.tile(frame, (max(1, c["obs_history"]), 1))
        obs = {"state": s["hist"].reshape(-1), "privileged_state": self._critic(d, s, mf)}
        z = jnp.zeros((), d.qpos.dtype)
        metrics = {n: z for n in ("vx", "x_abs", "failed", "clipped", "v_track_err")}
        metrics.update({f"r_{t.name}": z for t in rewards.TERMS})
        return State(pipeline_state=d, obs=obs, reward=z, done=z,
                     metrics=metrics, info={"rig": s})

    def step(self, state: State, action: jax.Array) -> State:
        k, c = self.k, self.c_
        s = dict(state.info["rig"])
        d = state.pipeline_state
        mf = s["mf"]
        key, k_push, k_force, k_cmd, k_noise = jax.random.split(s["key"], 5)
        s["key"] = key
        action_raw = action
        action = jnp.clip(action, -1.0, 1.0)
        target_pull = jnp.clip(s["pull_bias"] + c["residual_scale"] * action[k.servo_of], 0.0, 1.0)
        target_ctrl = k.l0 - k.stroke * target_pull
        s["servo_cmd"] = jnp.clip(0.5 * s["servo_cmd"] + 0.5 * target_ctrl, k.ctrl_lo, k.l0)
        s["cmd_buf"] = jnp.roll(s["cmd_buf"], 1, axis=0).at[0].set(s["servo_cmd"])

        # random carriage pushes
        if c["dr_enabled"]:
            p = c["dr_push_prob"] * c["dr_scale"]
            start = (s["push_left"] == 0) & (jax.random.uniform(k_push) < p)
            f = c["dr_push_force"]
            s["push_force"] = jnp.where(start, jax.random.uniform(k_force, (), minval=-f, maxval=f),
                                        s["push_force"])
            s["push_left"] = jnp.where(start, c["dr_push_steps"], s["push_left"])
        push = jnp.where(s["push_left"] > 0, s["push_force"], 0.0)
        s["push_left"] = jnp.maximum(s["push_left"] - 1, 0)
        d = d.replace(xfrc_applied=d.xfrc_applied.at[k.slider_body, 0].set(push))

        # physics substeps: servo model then mjx.step
        speed = c["servo_speed"]
        step_dl = speed * k.timestep if speed > 0 else jnp.inf
        sent = s["cmd_buf"] + s["calib"]
        nsub = self.cfg.n_substeps
        delay = s["delay"]
        track_foot = c["impact_weight"] != 0.0 or c["slip_weight"] != 0.0
        r = k.foot_radius

        def substep(carry, kk):
            d, peak, f_peak, slip_sum, slip_n, volt = carry
            mf_k = mf
            if self.per_env_models:
                load = jnp.mean(jnp.clip(-d.actuator_force / k.fmax, 0.0, 1.0))
                volt = s["battery"] * (1.0 - s["sag"] * load)
                fr = mf["actuator_forcerange"].at[:, 0].set(k.fmax_nom * s["strength"] * volt)
                mf_k = {**mf, "actuator_forcerange": fr}
            lag = jnp.where(kk >= delay, 0, (delay - kk + nsub - 1) // nsub)
            target = sent[jnp.minimum(lag, CMD_BUF - 1)]
            max_dl = step_dl * volt
            ctrl = d.ctrl + jnp.clip(target - d.ctrl, -max_dl, max_dl)
            d = mjx.step(self._model(mf_k), d.replace(ctrl=ctrl))
            peak = jnp.maximum(peak, -d.actuator_force)
            if track_foot:
                sd = d.sensordata
                fn = sd[k.foot_force_adr]
                slip = jnp.hypot(sd[k.foot_linvel_adr] - r * sd[k.foot_angvel_adr + 1],
                                 sd[k.foot_linvel_adr + 1] + r * sd[k.foot_angvel_adr])
                f_peak = jnp.maximum(f_peak, fn)
                touch = fn > 1e-6
                slip_sum = slip_sum + jnp.where(touch, slip, 0.0)
                slip_n = slip_n + touch
            return (d, peak, f_peak, slip_sum, slip_n, volt), None

        f0 = jnp.zeros((), dtype=d.qpos.dtype)
        init = (d, jnp.full(N_TENDON, -jnp.inf, dtype=d.qpos.dtype), f0, f0, jnp.int32(0), f0 + 1.0)
        (d, peak, f_peak, slip_sum, slip_n, volt), _ = jax.lax.scan(substep, init, jnp.arange(nsub))
        s["volt"] = volt

        # command, reward, failure
        s["t"] = s["t"] + 1
        s["total_steps"] = s["total_steps"] + 1
        s = self._set_cmd(s, k_cmd)
        vx = d.qvel[0]
        tau = c["vel_avg_tau"]
        s["vx_avg"] = s["vx_avg"] + min(1.0, self.cfg.control_dt / tau) * (vx - s["vx_avg"]) if tau > 0 else vx
        tau_i = c["idle_tau"]
        s["vx_idle"] = s["vx_idle"] + (min(1.0, self.cfg.control_dt / tau_i) if tau_i > 0 else 1.0) * (vx - s["vx_idle"])
        quat = d.qpos[k.ball_qadr[:, None] + np.arange(4)]
        bend = 2.0 * jnp.arccos(jnp.clip(jnp.abs(quat[:, 0]), 0.0, 1.0))
        tip = d.site_xpos[k.tip_id]
        x = d.qpos[k.slide_qadr]
        slip = jnp.where(slip_n > 0, slip_sum / jnp.maximum(slip_n, 1), 0.0)
        st = mjx_rewards.StepState(
            t=s["t"], cmd=s["cmd"], vx=vx, v_track=s["vx_avg"], v_idle=s["vx_idle"], x=x,
            foot_z=tip[2], foot_y=tip[1], bend=bend,
            pull=(k.l0 - d.ctrl) / k.stroke, force=-d.actuator_force / k.fmax,
            tension=-d.actuator_force, prev_tension=s["prev_tension"], peak_tension=peak,
            fmax=k.fmax, action=action, action_raw=action_raw, prev_action=s["prev_action"],
            prev_action2=s["prev_action2"], foot_force_peak=f_peak, slip=slip)
        # the CPU env computes k from the steps taken before this one
        kcur = c["curriculum_k0"] ** (c["curriculum_kd"] ** ((s["total_steps"] - 1) / max(c["curriculum_iter_steps"], 1)))
        reward, parts, clipped = mjx_rewards.total(st, c, kcur)
        failed = ((tip[2] > c["foot_lift_limit"]) | (jnp.max(bend) > c["bend_limit"])
                  | (jnp.abs(x) > c["rail_half"] * 0.98) | ~jnp.all(jnp.isfinite(d.qpos)))
        reward = reward - jnp.where(failed, c["failure_penalty"], 0.0)

        s["prev_action2"] = s["prev_action"]
        s["prev_action"] = action
        s["prev_tension"] = -d.actuator_force
        frame = self._hw_frame(d, s, k_noise)
        s["hist"] = jnp.concatenate([s["hist"][1:], frame[None]], axis=0)
        obs = {"state": s["hist"].reshape(-1), "privileged_state": self._critic(d, s, mf)}
        fdt = d.qpos.dtype
        metrics = {"vx": vx, "x_abs": jnp.abs(x), "failed": failed.astype(fdt),
                   "clipped": clipped.astype(fdt), "v_track_err": jnp.abs(s["vx_avg"] - s["cmd"]),
                   **{f"r_{n}": v for n, v in parts.items()}}
        done = jnp.maximum(state.done, failed.astype(fdt))
        return state.replace(pipeline_state=d, obs=obs, reward=reward, done=done,
                             metrics={**state.metrics, **metrics}, info={**state.info, "rig": s})


class ResampleOnDone:
    """Brax-style auto reset that starts a *new* episode (fresh bank draw and
    command) instead of replaying the first one.  Wraps a batched env."""

    def __init__(self, env, inner: MjxRigEnv):
        self.env, self.inner = env, inner

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state: State, action):
        if "steps" in state.info:
            state.info["steps"] = jnp.where(state.done, 0, state.info["steps"])
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        keys = jax.vmap(lambda k: jax.random.split(k)[0])(state.info["rig"]["key"])
        fresh = jax.vmap(self.inner.reset)(keys)

        def pick(a, b):
            m = state.done.reshape((-1,) + (1,) * (a.ndim - 1)) > 0
            return jnp.where(m, a, b)

        total = state.info["rig"]["total_steps"]          # the curriculum keeps counting
        rig = jax.tree.map(pick, fresh.info["rig"], state.info["rig"])
        rig["total_steps"] = total
        return state.replace(
            pipeline_state=jax.tree.map(pick, fresh.pipeline_state, state.pipeline_state),
            obs=jax.tree.map(pick, fresh.obs, state.obs),
            info={**state.info, "rig": rig})


def wrap_for_training(env: MjxRigEnv):
    """Brax ppo.train(wrap_env_fn=...) factory: vmap, episode limit, resample on done."""
    from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

    def wrap(e, episode_length, action_repeat, randomization_fn=None):
        w = VmapWrapper(e)
        w = EpisodeWrapper(w, episode_length, action_repeat)
        return ResampleOnDone(w, e)
    return wrap
