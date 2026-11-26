"""Reward for the single-leg rig: the terms, the presets (v1-v5), and overrides.

Read REWARD.md first if you are changing the reward.  In short:

    reward = sum over TERMS of  weight * k * term(state, cfg)
             (bounded costs capped so the step reward stays >= 0, see bound_costs)
             - failure_penalty   if the episode failed this step

* ``term(state, cfg)`` returns the raw value of one term: positive for something
  good, negative for a cost.  ``state`` is a ``StepState`` (everything a term may
  read for one env at one control step), ``cfg`` is the env's config dict.
* ``weight`` is the RigConfig field named in the Term (``--set <field>=<value>``).
* ``k`` is the cost curriculum factor for terms with ``curriculum=True`` (rises
  from ``curriculum_k0`` towards 1 during training), else 1.
* Terms with ``bounded=True`` may pull a step's reward down to 0 but not below,
  when ``bound_costs`` is on, so failing early never scores better than surviving.

Every term's weighted value is logged to TensorBoard as ``/r_<name>`` (mean
over envs); they sum to ``/reward_mean`` before the bounded-cost cap
(``/clipped`` = fraction of steps where it applied) and the failure penalty.

The reward only runs in training, so terms may use simulator-only quantities
(true carriage speed, foot contact force) that the real robot cannot measure.
"""

from __future__ import annotations

import ast
import difflib
import math
from dataclasses import dataclass, fields
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
# What a term can read
# --------------------------------------------------------------------------- #
@dataclass
class StepState:
    """One env, one control step, after the physics step.  Units are SI."""
    t: int                     # control steps since reset
    cmd: float                 # commanded carriage speed (m/s, signed)
    vx: float                  # instantaneous carriage speed (m/s)
    v_track: float             # speed the velocity terms score (vx, or its ~1 s average under v4)
    v_idle: float              # vx averaged over idle_tau (~0.5 s), for the idle term
    x: float                   # carriage position on the rail (m, 0 = centre)
    foot_z: float              # foot centre height (m); foot radius is FOOT_RADIUS
    foot_y: float              # foot sideways offset from the rail line (m)
    bend: np.ndarray           # (13,) bend angle of each ball joint (rad)
    pull: np.ndarray           # (8,) servo setpoint as tendon pull, 0..1
    force: np.ndarray          # (8,) tendon tension / max servo force
    tension: np.ndarray        # (8,) tendon tension now (N)
    prev_tension: np.ndarray   # (8,) tendon tension one control step ago (N)
    peak_tension: np.ndarray   # (8,) highest tension during this control step (N)
    fmax: np.ndarray           # (8,) nominal max servo force (N)
    action: np.ndarray         # (8,) policy output clipped to [-1, 1]
    action_raw: np.ndarray     # (8,) policy output before clipping
    prev_action: np.ndarray    # (8,) clipped action one step ago
    prev_action2: np.ndarray   # (8,) clipped action two steps ago
    foot_force_peak: float     # highest floor force on the foot during this step (N)
    slip: float                # mean foot sliding speed while in contact this step (m/s)


FOOT_RADIUS = 0.015            # m, foot sphere radius (rig_env re-exports this)


# --------------------------------------------------------------------------- #
# The terms (raw values; weights are applied in total())
# --------------------------------------------------------------------------- #
def vel(s: StepState, c: dict) -> float:
    """Speed tracking, 0..1: a bell curve of (v_track - cmd) with width vel_sigma."""
    sigma = max(c["vel_sigma"], 1e-6)
    return math.exp(-0.5 * ((s.v_track - s.cmd) / sigma) ** 2)


def vel_l2(s: StepState, c: dict) -> float:
    """Squared speed error in units of vel_sigma (keeps a gradient far from the target)."""
    sigma = max(c["vel_sigma"], 1e-6)
    return -(((s.v_track - s.cmd) / sigma) ** 2)


def prog(s: StepState, c: dict) -> float:
    """Progress: v_track / cmd, clipped to 0..1.5 (paid for moving the commanded way).

    Divides by at least cmd_min: with a ramped command, cmd passes near 0 at
    every reversal and vx / cmd would pay the maximum for drift.
    """
    if abs(s.cmd) <= 1e-6:
        return 0.0
    denom = math.copysign(max(abs(s.cmd), c["cmd_min"]), s.cmd)
    return float(np.clip(s.v_track / denom, 0.0, 1.5))


def idle(s: StepState, c: dict) -> float:
    """Idling: shortfall of the ~0.5 s average speed from the command, 0..1.

    0 when the carriage keeps up with the command over the last idle_tau
    seconds, 1 when it stands still or goes the wrong way.  Without it, a
    policy can push hard for a moment and then idle, and the 1 s average
    still scores close to the command.  Calibration (3 N rail, 0.07 m/s): the
    scripted walk and its clone average 0.22, policies that visibly idle
    0.47-0.57, a policy that stopped 0.99.
    """
    if abs(s.cmd) <= 1e-6:
        return 0.0
    denom = math.copysign(max(abs(s.cmd), c["cmd_min"]), s.cmd)
    return -float(np.clip(1.0 - s.v_idle / denom, 0.0, 1.0))


def rail(s: StepState, c: dict) -> float:
    """Near the rail ends: quadratic beyond 70 % of the half length."""
    half = c["rail_half"]
    over = max(0.0, abs(s.x) - 0.7 * half)
    return -0.5 * (over / (0.3 * half)) ** 2


def foot(s: StepState, c: dict) -> float:
    """Foot lift above the floor, beyond foot_free metres (2 cm costs 0.5)."""
    if c["foot_free"] > 0:
        lift = max(0.0, s.foot_z - FOOT_RADIUS - c["foot_free"])
    else:
        lift = s.foot_z - FOOT_RADIUS
    return -0.5 * (lift / 0.02) ** 2


def lat(s: StepState, c: dict) -> float:
    """Foot sideways offset beyond lat_free metres (3 cm costs 0.5)."""
    side = max(0.0, abs(s.foot_y) - c["lat_free"])
    return -0.5 * (side / 0.03) ** 2


def effort(s: StepState, c: dict) -> float:
    """Mean squared tendon pull."""
    return -float(np.mean(s.pull ** 2))


def rate(s: StepState, c: dict) -> float:
    """Action change: mean (a_t - a_{t-1})^2."""
    return -float(np.mean((s.action - s.prev_action) ** 2))


def ten(s: StepState, c: dict) -> float:
    """Mean tendon tension (fraction of max)."""
    return -float(np.mean(s.force))


def action_l2(s: StepState, c: dict) -> float:
    """Raw (un-clipped) action size.  Without it the policy drifts into |a| > 5
    and clipping removes all command-magnitude information."""
    return -float(np.mean(s.action_raw ** 2))


def accel(s: StepState, c: dict) -> float:
    """Jerk: mean (a_t - 2 a_{t-1} + a_{t-2})^2, sudden reversals of the servo targets."""
    if s.t < 3:
        return 0.0
    a2 = s.action - 2.0 * s.prev_action + s.prev_action2
    return -float(np.mean(a2 ** 2))


def shock(s: StepState, c: dict) -> float:
    """Tension shock: change in tension over one step, 10 % of max force (20 N) = 1."""
    d_ten = (s.tension - s.prev_tension) / (0.1 * s.fmax)
    return -float(np.mean(d_ten ** 2))


def tlim(s: StepState, c: dict) -> float:
    """Soft tension limit: 0 below tension_soft x max, 1 per cable at the max."""
    soft = c["tension_soft"]
    over = np.clip(s.peak_tension / s.fmax - soft, 0.0, None) / (1.0 - soft)
    return -float(np.sum(over ** 2))


def blim(s: StepState, c: dict) -> float:
    """Soft bend limit: 0 below bend_soft x bend_limit, 1 per joint at the limit."""
    soft = c["bend_soft"]
    over = np.clip(s.bend / c["bend_limit"] - soft, 0.0, None) / (1.0 - soft)
    return -float(np.sum(over ** 2))


def impact(s: StepState, c: dict) -> float:
    """Foot impact: peak foot force above 2x the standing load, in standing loads."""
    f_static = c["foot_static_force"]
    over = max(0.0, s.foot_force_peak - 2.0 * f_static) / f_static
    return -(over ** 2)


def slip(s: StepState, c: dict) -> float:
    """Foot sliding while planted, slip_scale m/s = 1, capped at slip_cap.  Linear:
    the squared form let one skid outweigh the failure penalty."""
    return -min(s.slip / c["slip_scale"], c["slip_cap"])


@dataclass(frozen=True)
class Term:
    name: str                              # logged as /r_<name>
    fn: Callable[[StepState, dict], float]
    weight: str                            # RigConfig field holding the weight
    curriculum: bool = False               # multiply by the cost curriculum k
    bounded: bool = False                  # capped by bound_costs


# Order matters only for float rounding (sums run in this order).
TERMS: tuple[Term, ...] = (
    # v1 terms
    Term("vel", vel, "vel_weight"),
    Term("vel_l2", vel_l2, "vel_l2_weight"),
    Term("prog", prog, "progress_weight"),
    Term("rail", rail, "rail_weight"),
    Term("foot", foot, "foot_weight"),
    Term("lat", lat, "lat_weight"),
    Term("effort", effort, "effort_weight"),
    Term("rate", rate, "rate_weight", curriculum=True, bounded=True),
    Term("ten", ten, "tension_weight"),
    Term("action_l2", action_l2, "action_l2_weight"),
    # v2 smoothness and damage costs
    Term("accel", accel, "accel_weight", curriculum=True, bounded=True),
    Term("shock", shock, "shock_weight", curriculum=True, bounded=True),
    Term("tlim", tlim, "tension_limit_weight", curriculum=True, bounded=True),
    Term("blim", blim, "bend_limit_weight", curriculum=True, bounded=True),
    Term("impact", impact, "impact_weight", curriculum=True, bounded=True),
    Term("slip", slip, "slip_weight", curriculum=True, bounded=True),
    # v5
    Term("idle", idle, "idle_weight"),
)


# Fields that change only the reward (not the physics, the command, the
# observations or when an episode ends).  reward_check.lockstep varies only
# these between its two envs, so both see exactly the same motion.
REWARD_FIELDS: frozenset[str] = frozenset({t.weight for t in TERMS} | {
    "vel_sigma", "vel_avg_tau", "idle_tau", "lat_free", "foot_free", "tension_soft", "bend_soft",
    "foot_static_force", "slip_scale", "slip_cap", "bound_costs",
    "curriculum_k0", "curriculum_kd", "failure_penalty", "reward_preset",
})

def total(s: StepState, c: dict, k: float) -> tuple[float, dict, bool]:
    """Step reward before the failure penalty, each term's weighted value, and
    whether the bounded costs were capped this step."""
    parts = {}
    for term in TERMS:
        w = c[term.weight] * k if term.curriculum else c[term.weight]
        parts[term.name] = w * term.fn(s, c)
    reward = 0.0
    for value in parts.values():
        reward += value
    clipped = False
    if c["bound_costs"]:
        costs = 0.0
        for term in TERMS:
            if term.bounded:
                costs += parts[term.name]
        costs = -costs
        base = reward + costs
        applied = min(costs, max(base, 0.0))
        clipped = applied < costs
        reward = base - applied
    return reward, parts, clipped


# --------------------------------------------------------------------------- #
# Presets: only what differs from the RigConfig defaults (which are v2)
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict] = {
    # The original reward: step command, tiny rate cost, no curriculum, no v2 costs.
    "v1": {"cmd_ramp": 0.0, "rate_weight": 0.02, "curriculum_k0": 1.0, "bound_costs": False,
           "accel_weight": 0.0, "shock_weight": 0.0, "tension_limit_weight": 0.0,
           "bend_limit_weight": 0.0, "impact_weight": 0.0, "slip_weight": 0.0},
    # v2: ramped command + smoothness/damage costs with a curriculum (the defaults).
    "v2": {},
    # v3: the foot may lift 3 cm and swing 10 cm sideways for free (the swing
    # back has to go around the side; v1/v2 penalised that path).
    "v3": {"lat_free": 0.10, "foot_free": 0.03},
    # v4: score the carriage speed averaged over ~one stride (1 s); one leg moves
    # the carriage in pulses, and scoring the instantaneous speed ranks every
    # gait below standing still.  vel_sigma 0.03 is the value v4 was tested with.
    "v4": {"lat_free": 0.10, "foot_free": 0.03, "vel_avg_tau": 1.0, "vel_sigma": 0.03},
    # v5: v4 + a cost for idling (pushing hard, then standing still), which
    # v4's 1 s average let KL fine-tunes of the clone drift into.
    "v5": {"lat_free": 0.10, "foot_free": 0.03, "vel_avg_tau": 1.0, "vel_sigma": 0.03,
           "idle_weight": 1.0, "idle_tau": 0.5},
}
DEFAULT_PRESET = "v5"


def apply_preset(cfg, name: str):
    """Set cfg's reward fields to preset ``name`` (other fields untouched)."""
    if name not in PRESETS:
        raise ValueError(f"unknown reward preset {name!r}; choose from {sorted(PRESETS)}")
    for key, value in PRESETS[name].items():
        if not hasattr(cfg, key):
            raise AttributeError(f"preset {name} sets unknown RigConfig field {key!r}")
        setattr(cfg, key, value)
    cfg.reward_preset = name
    return cfg


def apply_overrides(cfg, pairs):
    """Apply ``["field=value", ...]`` to cfg (any RigConfig field, not only rewards).

    Values are parsed by the field's current type: numbers, true/false,
    strings, and Python literals for tuples / dicts (``dr_friction=(0.3,1.0)``).
    """
    names = [f.name for f in fields(cfg)]
    for pair in pairs or ():
        if "=" not in pair:
            raise SystemExit(f"--set expects field=value, got {pair!r}")
        key, text = (part.strip() for part in pair.split("=", 1))
        if key not in names:
            close = difflib.get_close_matches(key, names, n=3)
            raise SystemExit(f"--set: no RigConfig field {key!r}"
                             + (f" (did you mean {', '.join(close)}?)" if close else ""))
        current = getattr(cfg, key)
        if isinstance(current, bool):
            if text.lower() not in ("true", "false", "1", "0"):
                raise SystemExit(f"--set {key}: expected true/false, got {text!r}")
            value = text.lower() in ("true", "1")
        elif isinstance(current, int):
            value = int(text)
        elif isinstance(current, float):
            value = float(text)
        elif isinstance(current, str):
            value = text
        else:
            value = ast.literal_eval(text)
        setattr(cfg, key, value)
    return cfg
