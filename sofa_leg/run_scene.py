"""runSofa entry point: the leg with its STL disks, driven in a circle or by the keyboard.

From an activated `sofa` conda env (so runSofa finds Python 3.12 and SOFA_ROOT):
  & "$env:SOFA_ROOT\\bin\\runSofa.exe" -l SofaPython3 sofa_leg\\run_scene.py
  $env:SOFA_LEG_XML = "path\\to\\other\\leg.xml"     # optional, default urdf/gradus/sim/leg.xml

Press Animate (or space in the viewer) to start. Keys (the 3D view must have focus; some SOFA GUIs only pass keys
while Ctrl is held):
  T            switch between the circle drive and keyboard teleop
  arrow keys   teleop: pull the front/back/left/right inner tendon harder (each press +10% of the stroke)
  O            teleop: also pull the outer tendons, at 45% of the inner pull (as the circle drive does)
  R            teleop: release every tendon
Headless use, data and validation live in sofa_leg.play / sofa_leg.collect / sofa_leg.validate.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import Sofa  # noqa: E402
import Sofa.Core  # noqa: E402

from genesis_leg.leg_play import OUTER_SCALE, circle_pull  # noqa: E402
from sofa_leg.scene import build_leg  # noqa: E402
from sofa_leg.spec import RodParams, leg_spec  # noqa: E402

AMPLITUDE, HZ, STEP = 0.5, 0.35, 0.1
ARROWS = {"up": "front", "down": "back", "left": "left", "right": "right"}
KEY_CODES = {19: "up", 21: "down", 18: "left", 20: "right"}  # Qt-style codes some SOFA GUIs send for arrows


class DriveController(Sofa.Core.Controller):
    def __init__(self, handle, spec, **kwargs):
        super().__init__(**kwargs)
        self.handle, self.spec = handle, spec
        self.names = [t.name for t in spec.tendons]
        self.low = np.array([t.ctrl_low for t in spec.tendons])
        self.high = np.array([t.ctrl_high for t in spec.tendons])
        self.teleop, self.outer, self.time = False, False, 0.0
        self.pull = np.zeros(len(self.names))

    def onAnimateBeginEvent(self, event):
        if self.teleop:
            pull = self.pull.copy()
            if self.outer:
                for i, name in enumerate(self.names):
                    if "_outer_" in name:
                        pull[i] = OUTER_SCALE * self.pull[self.names.index(name.replace("_outer_", "_inner_"))]
        else:
            pull = np.array(circle_pull(self.names, self.time, AMPLITUDE, HZ))
        self.handle.servo.ctrl = self.high - np.clip(pull, 0, 1) * (self.high - self.low)
        self.handle.servo.update()  # the servo may already have run this step; apply the new targets now
        self.time += self.handle.dt

    def onKeypressedEvent(self, event):
        key = event.get("key")
        key = KEY_CODES.get(key, key) if isinstance(key, int) else str(key).lower()
        if key == "t":
            self.teleop = not self.teleop
            print("teleop" if self.teleop else "circle drive")
        elif key == "o":
            self.outer = not self.outer
        elif key == "r":
            self.pull[:] = 0
        elif key in ARROWS:
            name = f"tendon_inner_{ARROWS[key]}"
            if name in self.names:
                i = self.names.index(name)
                self.pull[i] = min(1.0, self.pull[i] + STEP)


def createScene(root):
    xml = os.environ.get("SOFA_LEG_XML", str(REPO / "urdf" / "gradus" / "sim" / "leg.xml"))
    spec = leg_spec(xml, RodParams())
    handle = build_leg(root, spec, visual=True)
    root.addObject(DriveController(handle, spec, name="drive"))
    return root
