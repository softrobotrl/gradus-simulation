"""SofaLeg: the leg in SOFA behind the same small interface as genesis_leg.leg_validate.MujocoLeg.

    leg = SofaLeg("urdf/gradus/sim/leg.xml")
    leg.reset()
    leg.set_pull([0.5, 0, 0, 0, 0, 0, 0, 0])   # fraction of each servo's stroke, 0 = rest, 1 = fully pulled
    leg.step(1000)                             # 1000 x dt
    leg.tip(), leg.tension(), leg.length()

One SofaLeg is one SOFA simulation (SOFA does not batch environments); run several processes for throughput.
"""

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

from sofa_leg.scene import build_leg, quat_rotate
from sofa_leg.spec import RodParams, leg_spec

DEFAULT_DT = 0.001  # s; 2 ms drifts 2.5% of the path from a 0.5 ms reference over 0.4 s, 1 ms drifts 0.9%


class SofaLeg:
    def __init__(self, xml, rod=None, dt=DEFAULT_DT, rayleigh=None, gravity=None, visual=False,
                 linear_solver="SparseLDLSolver"):
        self.spec = leg_spec(xml, rod or RodParams())
        if gravity is not None:
            self.spec.gravity = np.asarray(gravity, dtype=float)
        self.dt = dt
        self.root = Sofa.Core.Node("root")
        self.handle = build_leg(self.root, self.spec, dt=dt, rayleigh=rayleigh, visual=visual,
                                linear_solver=linear_solver)
        Sofa.Simulation.init(self.root)
        self.tendon_names = [t.name for t in self.spec.tendons]
        self.ctrl_low = np.array([t.ctrl_low for t in self.spec.tendons])
        self.ctrl_high = np.array([t.ctrl_high for t in self.spec.tendons])
        self.time = 0.0

    @property
    def servo(self):
        return self.handle.servo

    @property
    def n_free_nodes(self):
        return len(self.handle.free.dofs.position.value)

    @property
    def n_rod_nodes(self):
        return len(self.handle.nodes)

    def reset(self):
        """Back to the straight starting pose at rest, servos at rest length."""
        Sofa.Simulation.reset(self.root)
        self.servo.ctrl = self.ctrl_high.copy()
        self.servo.update()
        self.time = 0.0

    def set_pull(self, pull):
        """Servo targets from pull fractions in [0, 1], exactly as MujocoLeg.set_pull."""
        self.servo.ctrl = self.ctrl_high - np.clip(pull, 0, 1) * (self.ctrl_high - self.ctrl_low)

    def set_ctrl(self, ctrl):
        self.servo.ctrl = np.asarray(ctrl, dtype=float).copy()

    def step(self, n=1):
        for _ in range(n):
            Sofa.Simulation.animate(self.root, self.dt)
        self.time += n * self.dt
        self.servo.update()  # report tension/length for the state the step ended in

    # ---- readouts ---------------------------------------------------------------------------------------------
    def disk_frames(self):
        """(n_disks, 7) world position + quaternion (x y z w) of each disk's centre of mass."""
        return np.asarray(self.handle.disks.dofs.position.value).copy()

    def disk_velocities(self):
        """(n_disks, 6) linear velocity of the centre of mass, then angular velocity, world frame."""
        return np.asarray(self.handle.disks.dofs.velocity.value).copy()

    def point(self, disk, local):
        frame = self.disk_frames()[disk]
        return frame[:3] + quat_rotate(frame[3:], np.asarray(local, dtype=float))

    def point_velocity(self, disk, local):
        frame, vel = self.disk_frames()[disk], self.disk_velocities()[disk]
        return vel[:3] + np.cross(vel[3:], quat_rotate(frame[3:], np.asarray(local, dtype=float)))

    def tip(self, *_):
        """Foot tip in world coordinates (MujocoLeg.tip takes (body, local); SofaLeg already knows them)."""
        return self.point(self.spec.tip_disk, self.spec.tip_local)

    def tip_velocity(self):
        return self.point_velocity(self.spec.tip_disk, self.spec.tip_local)

    def tension(self):
        return self.servo.tension.copy()

    def length(self):
        return self.servo.length.copy()

    def length_rate(self):
        return self.servo.length_rate.copy()

    def rod_nodes(self):
        return np.asarray(self.handle.rod.dofs.position.value).copy()

    def finite(self):
        return bool(np.isfinite(self.disk_frames()).all() and np.isfinite(self.disk_velocities()).all())

    # ---- whole-state save/restore (for data collection from a fixed start state) -------------------------------
    STATE_FIELDS = ("position", "velocity", "free_position", "free_velocity", "force", "externalForce")

    def _mechanical_objects(self, node=None, path=""):
        node = node or self.root
        for obj in node.objects:
            if obj.getClassName() == "MechanicalObject":
                yield f"{path}/{obj.name.value}", obj
        for child in node.children:
            # a node with several parents (the rod) is visited once per parent; the path keeps the first
            yield from self._mechanical_objects(child, f"{path}/{child.name.value}")

    def get_state(self):
        """Every MechanicalObject's vectors (mapped ones too) plus the servo targets, so a restore is exact."""
        state = {"ctrl": self.servo.ctrl.copy(), "time": self.time}
        for path, mo in self._mechanical_objects():
            for name in self.STATE_FIELDS:
                data = mo.getData(name)
                if data is not None:
                    state[f"{path}:{name}"] = np.array(data.value, dtype=float, copy=True)
        return state

    def set_state(self, state, flush=True):
        """Restore a get_state() snapshot.

        SOFA keeps a little solver state outside the MechanicalObjects (it carries over from the previous step), so
        a bare restore lands up to ~0.2 mm away on the foot after 60 steps depending on what ran before. With
        flush=True the snapshot is restored, stepped once, and restored again, which makes that carried-over state
        the same every time: repeated restores then agree to ~1e-7 m whatever ran in between.
        """
        self._write_state(state)
        if flush:
            self.step(1)
            self._write_state(state)

    def _write_state(self, state):
        for path, mo in self._mechanical_objects():
            for name in self.STATE_FIELDS:
                key = f"{path}:{name}"
                if key in state and len(state[key]):
                    mo.getData(name).value = state[key].tolist()
        self.servo.ctrl = state["ctrl"].copy()
        self.time = state["time"]
        self.servo.update()
