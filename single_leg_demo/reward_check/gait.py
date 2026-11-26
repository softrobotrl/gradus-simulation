"""Scripted side-swing walk, expressed as env actions (same clip / scale / filter as a policy).

Not a controller for the real leg: a fixed, known-good behaviour for testing
rewards.  Pull order: IF IL IB IR OF OL OB OR (see rig_env).  Mirrors
front <-> back for a negative command.

Cycle: plant in front -> push with both back cables (the foot lifts off at the
end of the stroke) -> bend sideways (lifts the foot) -> swing forward at the
side -> release the side cable (replant).  The foot touches the floor anywhere
within ~4 cm of under the hip, so the swing back has to go around the side.
"""

from __future__ import annotations

import numpy as np


def P(IF=0, IL=0, IB=0, IR=0, OF=0, OL=0, OB=0, OR=0):
    return np.array([IF, IL, IB, IR, OF, OL, OB, OR], float)


PLANT = P(OF=.75, OB=.25)
PUSH = P(IB=1, OB=1)
SIDE = P(IL=1)
KEYS = [PLANT, PUSH, PUSH + SIDE, PLANT + SIDE]
MIRROR = [2, 1, 0, 3, 6, 5, 4, 7]


class ScriptedGait:
    """Keyframed tendon pulls, smoothstep-interpolated.

    seg_steps: control steps per keyframe segment; None = slower for a slower
    command.  ``jerky = True`` jumps between keyframes instead of easing.
    """

    X_STOP = 0.28   # m: beyond this, in the direction of travel, hold the planted pose

    def __init__(self, n: int, seg_steps: int | None = None, k_speed: float = 0.9):
        self.fixed, self.k_speed = seg_steps, k_speed
        self.k = np.zeros(n, int)
        self.sign = np.zeros(n)
        self.seg = np.full(n, 8)
        self.start = [None] * n
        self.jerky = False

    def reset(self, i: int) -> None:
        self.sign[i] = 0

    def pull(self, i: int, cmd: float, mag: float, cur_pull: np.ndarray) -> np.ndarray:
        sgn = 1.0 if cmd >= 0 else -1.0
        if sgn != self.sign[i]:
            self.sign[i] = sgn
            self.k[i] = 0
            self.start[i] = cur_pull.copy()
            self.seg[i] = self.fixed or int(np.clip(round(self.k_speed / max(mag, 1e-3)), 8, 20))
        n = self.seg[i]
        seg, within = divmod(self.k[i], n)
        keys = [kf if sgn > 0 else kf[MIRROR] for kf in KEYS]
        a = self.start[i] if seg == 0 else keys[(seg - 1) % 4]
        b = keys[seg % 4]
        s = (within + 1) / n
        s = 1.0 if self.jerky else s * s * (3 - 2 * s)
        self.k[i] += 1
        return np.clip((1 - s) * a + s * b, 0, 1)

    def action(self, env, i: int) -> np.ndarray:
        """Action in [-1, 1] for env i that produces the scripted pull."""
        cur = np.clip((env.l0 - env.datas[i].ctrl) / env.stroke, 0, 1)
        cmd = float(env.cmd[i])
        x = float(env.datas[i].qpos[env.slide_qadr])
        if (self.sign[i] != 0 and (1.0 if cmd >= 0 else -1.0) == self.sign[i]
                and self.sign[i] * x > self.X_STOP):
            hold = PLANT if self.sign[i] > 0 else PLANT[MIRROR]
            p = np.clip(0.7 * cur + 0.3 * hold, 0, 1)        # ease into the plant pose and wait
        else:
            p = self.pull(i, cmd, float(env.cmd_mag[i]), cur)
        a = np.clip((p - env.pull_bias[i]) / env.cfg["residual_scale"], -1, 1)
        if env.num_actions != len(a):        # fewer servos than tendons: average each spool's cables
            a = np.array([a[env.servo_of == s].mean() for s in range(env.num_actions)])
        return a


class SagittalGait:
    """Forward/back-only walk for the confirmed 4-servo leg (servo_layout="paired_x").

    One sine per servo (inner +X, inner -X, outer +X, outer -X), found by a
    random search over periodic patterns and checked on nominal physics
    (2026-11-26, logs/reward_v2_study/hw_v2): about 0.048 m/s both ways on the
    3 N rail, with about one foot lift per stride (1.22 Hz).  Open loop, so it
    does not cope with randomized physics; it is a starting point to clone.

    The pattern walks -X as found; a positive command swaps each ring's +X and
    -X servos.  Beyond X_STOP in the direction of travel it holds still.
    """

    FREQ = 1.2156                                   # strides per second
    AMP = np.array([0.984, 0.660, 0.987, 0.870])
    OFF = np.array([0.468, -0.436, 0.260, -0.271])
    PH = np.array([4.820, 2.284, 4.951, 3.857])
    MIRROR = [1, 0, 3, 2]
    X_STOP = 0.28

    def __init__(self, n: int):
        self.k = np.zeros(n, int)
        self.jerky = False                          # interface parity with ScriptedGait

    def reset(self, i: int) -> None:
        self.k[i] = 0

    def action(self, env, i: int) -> np.ndarray:
        if env.num_actions != 4:
            raise ValueError("SagittalGait needs the 4-servo leg (servo_layout='paired_x')")
        cmd = float(env.cmd[i])
        sgn = 1.0 if cmd >= 0 else -1.0
        x = float(env.datas[i].qpos[env.slide_qadr])
        t = self.k[i] * env.dt
        self.k[i] += 1
        if sgn * x > self.X_STOP:
            return np.zeros(4)
        a = np.clip(self.OFF + self.AMP * np.sin(2 * np.pi * self.FREQ * t + self.PH), -1, 1)
        return a[self.MIRROR] if sgn > 0 else a
