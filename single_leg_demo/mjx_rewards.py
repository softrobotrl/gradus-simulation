"""JAX copy of the reward terms in rewards.py, for the MJX env.

rewards.py stays the place to read and change the reward: term names, which
weight field each uses, curriculum/bounded flags and presets all come from
rewards.TERMS.  Only the formulas are repeated here, in jax.numpy (no Python
`if` on values).  tests/test_mjx_env.py checks every term here against
rewards.py on random states; a term missing here is an error.

When you add or change a term in rewards.py, mirror it below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from single_leg_demo import rewards

R = rewards.FOOT_RADIUS


@dataclass
class StepState:
    """Same fields as rewards.StepState, as JAX arrays (one env)."""
    t: Any
    cmd: Any
    vx: Any
    v_track: Any
    v_idle: Any
    x: Any
    foot_z: Any
    foot_y: Any
    bend: Any
    pull: Any
    force: Any
    tension: Any
    prev_tension: Any
    peak_tension: Any
    fmax: Any
    action: Any
    action_raw: Any
    prev_action: Any
    prev_action2: Any
    foot_force_peak: Any
    slip: Any


def _denom(s, c):
    return jnp.where(s.cmd >= 0, 1.0, -1.0) * jnp.maximum(jnp.abs(s.cmd), c["cmd_min"])


def vel(s, c):
    sigma = max(c["vel_sigma"], 1e-6)
    return jnp.exp(-0.5 * ((s.v_track - s.cmd) / sigma) ** 2)


def vel_l2(s, c):
    sigma = max(c["vel_sigma"], 1e-6)
    return -(((s.v_track - s.cmd) / sigma) ** 2)


def prog(s, c):
    return jnp.where(jnp.abs(s.cmd) <= 1e-6, 0.0, jnp.clip(s.v_track / _denom(s, c), 0.0, 1.5))


def idle(s, c):
    return jnp.where(jnp.abs(s.cmd) <= 1e-6, 0.0, -jnp.clip(1.0 - s.v_idle / _denom(s, c), 0.0, 1.0))


def rail(s, c):
    half = c["rail_half"]
    over = jnp.maximum(0.0, jnp.abs(s.x) - 0.7 * half)
    return -0.5 * (over / (0.3 * half)) ** 2


def foot(s, c):
    if c["foot_free"] > 0:
        lift = jnp.maximum(0.0, s.foot_z - R - c["foot_free"])
    else:
        lift = s.foot_z - R
    return -0.5 * (lift / 0.02) ** 2


def lat(s, c):
    side = jnp.maximum(0.0, jnp.abs(s.foot_y) - c["lat_free"])
    return -0.5 * (side / 0.03) ** 2


def effort(s, c):
    return -jnp.mean(s.pull ** 2)


def rate(s, c):
    return -jnp.mean((s.action - s.prev_action) ** 2)


def ten(s, c):
    return -jnp.mean(s.force)


def action_l2(s, c):
    return -jnp.mean(s.action_raw ** 2)


def accel(s, c):
    a2 = s.action - 2.0 * s.prev_action + s.prev_action2
    return jnp.where(s.t < 3, 0.0, -jnp.mean(a2 ** 2))


def shock(s, c):
    d_ten = (s.tension - s.prev_tension) / (0.1 * s.fmax)
    return -jnp.mean(d_ten ** 2)


def tlim(s, c):
    soft = c["tension_soft"]
    over = jnp.clip(s.peak_tension / s.fmax - soft, 0.0, None) / (1.0 - soft)
    return -jnp.sum(over ** 2)


def blim(s, c):
    soft = c["bend_soft"]
    over = jnp.clip(s.bend / c["bend_limit"] - soft, 0.0, None) / (1.0 - soft)
    return -jnp.sum(over ** 2)


def impact(s, c):
    f_static = c["foot_static_force"]
    over = jnp.maximum(0.0, s.foot_force_peak - 2.0 * f_static) / f_static
    return -(over ** 2)


def slip(s, c):
    return -jnp.minimum(s.slip / c["slip_scale"], c["slip_cap"])


FNS = {f.__name__: f for f in (vel, vel_l2, prog, idle, rail, foot, lat, effort, rate, ten,
                               action_l2, accel, shock, tlim, blim, impact, slip)}
missing = [t.name for t in rewards.TERMS if t.name not in FNS]
if missing:
    raise ImportError(f"mjx_rewards.py has no JAX version of reward term(s) {missing}; "
                      "mirror them from rewards.py")


def total(s: StepState, c: dict, k):
    """Same as rewards.total: (reward before the failure penalty, weighted
    terms, whether the bounded costs were capped)."""
    parts = {}
    for term in rewards.TERMS:
        w = c[term.weight] * k if term.curriculum else c[term.weight]
        parts[term.name] = w * FNS[term.name](s, c)
    reward = 0.0
    for v in parts.values():
        reward = reward + v
    clipped = jnp.bool_(False)
    if c["bound_costs"]:
        costs = -sum(parts[t.name] for t in rewards.TERMS if t.bounded)
        base = reward + costs
        applied = jnp.minimum(costs, jnp.maximum(base, 0.0))
        clipped = applied < costs
        reward = base - applied
    return reward, parts, clipped
