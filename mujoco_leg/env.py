"""Batched MJX environment for the tip-up tendon leg.

The tip is free in Z; the reward only cares about the XY projection.  The
policy controls the eight servo tendon targets (tension-only) and sees a
short look-ahead window of the target curve.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from mujoco_leg import shapes
from mujoco_leg.config import Config


class EnvState(NamedTuple):
    data: mjx.Data
    shape: jnp.ndarray       # (P, 2) target curve in metres
    closed: jnp.ndarray      # scalar bool
    phase: jnp.ndarray       # scalar in [0, 1)
    step: jnp.ndarray        # scalar int
    prev_action: jnp.ndarray # (8,)
    prev_tip: jnp.ndarray    # (3,)
    key: jnp.ndarray


class TracerEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        m = mujoco.MjModel.from_xml_path(cfg.model_path)
        # No contacts: fixed base, tip-up drawing task, and mesh collision is
        # both unnecessary and far too expensive for MJX.
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.m = m
        self.mx = mjx.put_model(m)

        self.tip_id = m.site(cfg.tip_site).id
        self.l0 = jnp.asarray(m.actuator_ctrlrange[:, 1])
        self.stroke = jnp.asarray(m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0])
        self.fmax = jnp.asarray(-m.actuator_forcerange[:, 0])
        self.nu = m.nu
        self.nq = m.nq
        self.nv = m.nv
        self.njnt = m.njnt
        self.dt = float(m.opt.timestep) * cfg.n_substeps

        self.n_points = cfg.shape_points
        self.radius = cfg.shape_radius
        self.K = cfg.lookahead
        self.lookahead_dt = cfg.lookahead_dt
        self.period = cfg.period
        self.episode_steps = cfg.episode_steps
        self.n_substeps = cfg.n_substeps

        dummy, obs = self.reset_one(jax.random.PRNGKey(0))
        self.obs_dim = int(obs.shape[0])

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset_one(self, key: jax.Array):
        k_shape, k_q, k_v, k_warm = jax.random.split(key, 4)
        shape, closed = shapes.sample_shape(k_shape, self.n_points, self.radius)

        d = mjx.make_data(self.m)
        noise = jax.random.normal(k_q, (self.njnt, 3)) * self.cfg.init_qpos_noise
        q = jnp.concatenate([jnp.ones((self.njnt, 1)), noise], axis=1)
        q = q / jnp.linalg.norm(q, axis=1, keepdims=True)
        d = d.replace(
            qpos=q.reshape(-1),
            qvel=jax.random.normal(k_v, (self.nv,)) * self.cfg.init_qvel_noise,
            ctrl=self.l0,
        )

        def settle(data, _):
            return mjx.step(self.mx, data), None

        if self.cfg.warmup_steps > 0:
            d = jax.lax.scan(settle, d, None, length=self.cfg.warmup_steps)[0]
        tip = d.site_xpos[self.tip_id]

        st = EnvState(
            data=d,
            shape=shape,
            closed=closed,
            phase=jnp.array(0.0),
            step=jnp.array(0, dtype=jnp.int32),
            prev_action=jnp.zeros(self.nu),
            prev_tip=tip,
            key=k_warm,
        )
        return st, self.obs(st)

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def obs(self, st: EnvState) -> jnp.ndarray:
        d = st.data
        tip = d.site_xpos[self.tip_id]
        tip_vel = (tip - st.prev_tip) / self.dt
        window = shapes.target_window(
            st.shape, st.phase, st.closed,
            self.K, self.lookahead_dt, self.period, self.n_points,
        )
        ten = (d.ten_length - self.l0) / self.stroke
        force = -d.actuator_force / self.fmax
        phase = jnp.array([
            jnp.sin(2.0 * jnp.pi * st.phase),
            jnp.cos(2.0 * jnp.pi * st.phase),
        ])
        return jnp.concatenate([
            d.qpos,                      # 52
            d.qvel / 10.0,               # 39
            ten,                         # 8
            force,                       # 8
            tip[:2] / 0.05,              # 2
            tip_vel[:2] / 0.2,           # 2
            window.reshape(-1) / 0.05,   # 2*(K+1)
            phase,                       # 2
        ])

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def step_one(self, st: EnvState, action: jnp.ndarray, auto_reset: bool = True):
        cfg = self.cfg
        pull = 0.5 * (action + 1.0)
        target_ctrl = self.l0 - self.stroke * pull
        ctrl = 0.5 * st.data.ctrl + 0.5 * target_ctrl
        ctrl = jnp.clip(ctrl, self.l0 - self.stroke, self.l0)

        d = st.data.replace(ctrl=ctrl)

        def physics(data, _):
            return mjx.step(self.mx, data), None

        d = jax.lax.scan(physics, d, None, length=self.n_substeps)[0]

        tip = d.site_xpos[self.tip_id]
        tip_vel = (tip - st.prev_tip) / self.dt

        target = shapes.target_at(st.shape, st.phase, st.closed, self.n_points)
        eps = 1e-3
        tp = shapes.target_at(st.shape, (st.phase + eps) % 1.0, st.closed, self.n_points)
        tm = shapes.target_at(st.shape, (st.phase - eps) % 1.0, st.closed, self.n_points)
        target_vel = (tp - tm) / (2.0 * eps * self.period)

        err = tip[:2] - target
        dist = jnp.sqrt(jnp.sum(err ** 2))
        # Gaussian peak plus a linear distance term: the linear part keeps a
        # useful gradient while the tip is still far from the target.
        r_track = jnp.exp(-0.5 * (dist / cfg.track_sigma) ** 2) - 0.5 * dist / 0.05
        r_vel = -0.01 * jnp.sum((tip_vel[:2] - target_vel) ** 2) / (0.15 ** 2)
        r_smooth = -0.005 * jnp.mean((action - st.prev_action) ** 2)
        r_ten = -0.001 * jnp.mean(-d.actuator_force / self.fmax)
        reward = r_track + r_vel + r_smooth + r_ten

        phase = (st.phase + self.dt / self.period) % 1.0
        step = st.step + 1
        done = (
            (~jnp.isfinite(d.qpos).all())
            | (tip[2] < 0.15)
            | (tip[2] > 0.95)
            | (jnp.linalg.norm(d.qvel) > 300.0)
            | (step >= self.episode_steps)
        )

        new = EnvState(d, st.shape, st.closed, phase, step, action, tip, st.key)
        k_reset, k_next = jax.random.split(st.key)

        if auto_reset:
            def do_reset(_):
                return self.reset_one(k_reset)

            def no_reset(_):
                return new, self.obs(new)

            next_st, next_obs = jax.lax.cond(done, do_reset, no_reset, operand=None)
            next_st = next_st._replace(key=k_next)
        else:
            next_st, next_obs = new, self.obs(new)

        info = {
            "err": dist,
            "track": jnp.exp(-0.5 * (dist / cfg.track_sigma) ** 2),
            "tension": jnp.mean(-d.actuator_force / self.fmax),
        }
        return next_st, next_obs, reward, done, info

    # ------------------------------------------------------------------ #
    # vectorised wrappers
    # ------------------------------------------------------------------ #
    def vreset(self, keys):
        return jax.vmap(self.reset_one)(keys)

    def vstep(self, states, actions):
        return jax.vmap(self.step_one)(states, actions)

    def vobs(self, states):
        return jax.vmap(self.obs)(states)
