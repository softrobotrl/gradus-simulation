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
        return np.clip((p - env.pull_bias[i]) / env.cfg["residual_scale"], -1, 1)
