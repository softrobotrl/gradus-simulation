"""Build the Gradus dog MJCF from the Onshape concept dimensions.

The CAD lives in ``cad/gradus_dog.fs``.  This module turns the same dimensions
into a MuJoCo model with:

* a floating trunk (free joint),
* four planar two-link legs (passive hip and knee hinges),
* two antagonistic tendons per leg, eight total,
* one position-controlled drum/carriage per tendon (eight servos).

The tendon model follows the existing STL -> MJCF pipeline: a small carriage
slides along the first tendon segment and stands in for the drum.  One radian
of drum rotation pays out/in ``drum_radius`` metres of cable, and a limited
spatial tendon turns the carriage motion into tendon tension.

Run::

    python -m pipeline.dog_model                    # writes pipeline/build/gradus_dog/robot.xml
    python -m pipeline.dog_model --out /tmp/dog     # writes elsewhere

The generated XML is self-contained (it includes the ground plane) and can be
opened directly::

    python -m mujoco.viewer --mjcf=pipeline/build/gradus_dog/robot.xml
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

DEFAULT_OUT = Path(__file__).parent / "build" / "gradus_dog"

# Trunk centre height in the CAD (metres).  All body-relative coordinates below
# are expressed relative to this point.
TRUNK_Z = 0.215


@dataclass(frozen=True)
class LegSpec:
    """One planar leg, from the Onshape FeatureScript."""

    name: str
    hip: tuple[float, float, float]
    knee: tuple[float, float, float]
    foot: tuple[float, float, float]
    side: int  # +1 left (y > 0), -1 right
    front: bool


LEGS: tuple[LegSpec, ...] = (
    LegSpec("FL", (0.100, 0.105, 0.175), (0.140, 0.105, 0.115), (0.110, 0.105, 0.015), +1, True),
    LegSpec("FR", (0.100, -0.105, 0.175), (0.140, -0.105, 0.115), (0.110, -0.105, 0.015), -1, True),
    LegSpec("RL", (-0.100, 0.105, 0.175), (-0.140, 0.105, 0.115), (-0.110, 0.105, 0.015), +1, False),
    LegSpec("RR", (-0.100, -0.105, 0.175), (-0.140, -0.105, 0.115), (-0.110, -0.105, 0.015), -1, False),
)

# Per-tendon x offsets of the hip guide, knee guide and foot anchor, in metres.
# The CAD's original +-18 mm parallel paths gave a rank-1 tendon Jacobian (both
# tendons produced the same joint-torque direction), so the leg could not hold
# itself up.  This routing gives tendon 0 a hip moment and tendon 1 a knee
# moment, making the 2x2 map well conditioned.  Rear legs mirror the x offsets.
#
# The offsets are the moment arms of the tendon drive: the joint torque is the
# cable tension times the offset, so they set how much torque the drum can
# produce at the joint.  30 mm was enough to stand but not to walk: the knee
# tendon saturated at 22 N of the 35 N limit and the leg buckled mid-stride.
# 45 mm roughly doubles the usable torque per newton of cable and the open-loop
# trot walks at 0.12 m/s without falling (see pipeline/README.md).
TENDON_GUIDES = (
    (0.045, 0.000, 0.045),   # t0 (red): mostly hip
    (-0.045, 0.045, -0.045), # t1 (blue): mostly knee, opposite hip
)


@dataclass(frozen=True)
class DogModelConfig:
    """Geometry, masses and servo parameters.

    Values marked ASSUMED are not in the concept CAD; they are conservative
    starting points for simulation and should be replaced when the hardware
    team provides real parts.
    """

    # Tendon routing: per-tendon x offsets of the hip guide, the knee guide and
    # the foot anchor, in metres.  Front legs take them in +x, rear legs mirror
    # them.  These must match the CAD (cad/gradus_dog.fs).
    tendon_guides: tuple[tuple[float, float, float], ...] = TENDON_GUIDES

    trunk_mass: float = 3.0       # kg, shell + battery + electronics; ASSUMED
    thigh_mass: float = 0.12      # kg, per thigh; ASSUMED (PLA + insert)
    calf_mass: float = 0.08       # kg, per calf tube; ASSUMED
    foot_mass: float = 0.04       # kg, per foot sphere; ASSUMED
    carriage_mass: float = 0.005  # kg, per drum carriage proxy; ASSUMED
    drum_radius: float = 0.012    # m, CAD drum radius
    rotor_inertia: float = 1.0e-5 # kg*m^2 at the motor; ASSUMED
    gear_ratio: float = 2.0       # motor turns per drum turn; ASSUMED 2:1 winch gearbox
    motor_torque: float = 0.25    # N*m at the motor; ASSUMED
    motor_max_speed: float = 16.0 # rad/s at the motor; ASSUMED
    kp: float = 2.0               # position servo P gain (actuator units)
    kv: float = 0.05              # position servo D gain (actuator units)
    angle_range: tuple[float, float] = (-0.8, 2.0)  # rad at the drum
    joint_damping: float = 0.02   # N*m*s/rad, passive hip/knee
    joint_armature: float = 1.0e-4
    joint_frictionloss: float = 0.005
    joint_range: float = 1.4      # rad, symmetric for hip and knee
    tendon_limit_timeconst: float = 0.002  # s, cable elasticity
    tendon_slack: float = 0.0     # m, extra cable at zero drum angle
    timestep: float = 0.001
    contact_timeconst: float = 0.005
    ground_friction: float = 1.0

    @property
    def max_torque(self) -> float:
        """Torque the drum can deliver (N*m): the motor torque through the gearbox."""
        return self.motor_torque * self.gear_ratio

    @property
    def max_speed(self) -> float:
        """Drum speed limit (rad/s): the motor speed divided by the gearbox."""
        return self.motor_max_speed / self.gear_ratio

    @property
    def reflected_inertia(self) -> float:
        return self.rotor_inertia * self.gear_ratio**2 / self.drum_radius**2

    @property
    def max_tension(self) -> float:
        return self.max_torque / self.drum_radius


def f(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _sub(parent: ET.Element, tag: str, **attrs: object) -> ET.Element:
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _rel_hip(leg: LegSpec) -> np.ndarray:
    return np.array(leg.hip) - np.array([0.0, 0.0, TRUNK_Z])


def _knee_vec(leg: LegSpec) -> np.ndarray:
    return np.array(leg.knee) - np.array(leg.hip)


def _foot_vec(leg: LegSpec) -> np.ndarray:
    return np.array(leg.foot) - np.array(leg.knee)


def _leg_x_sign(leg: LegSpec) -> float:
    """Rear legs mirror the tendon guide offsets in x."""
    return 1.0 if leg.front else -1.0


def _drum_center(leg: LegSpec, hip_offset: float) -> np.ndarray:
    return np.array([leg.hip[0] + _leg_x_sign(leg) * hip_offset,
                     leg.hip[1] + leg.side * 0.030,
                     leg.hip[2] + 0.044 - TRUNK_Z])


def _hip_guide(leg: LegSpec, hip_offset: float) -> np.ndarray:
    return np.array([leg.hip[0] + _leg_x_sign(leg) * hip_offset,
                     leg.hip[1], leg.hip[2] - TRUNK_Z])


def _knee_guide(leg: LegSpec, knee_offset: float) -> np.ndarray:
    v = _knee_vec(leg)
    return np.array([v[0] + _leg_x_sign(leg) * knee_offset, 0.0, v[2]])


def _foot_anchor(leg: LegSpec, foot_offset: float) -> np.ndarray:
    v = _foot_vec(leg)
    return np.array([v[0] + _leg_x_sign(leg) * foot_offset, 0.0, v[2]])


def _slide_axis(leg: LegSpec, hip_offset: float) -> np.ndarray:
    """Unit axis from the hip guide toward the drum: positive carriage motion pulls cable."""
    v = _drum_center(leg, hip_offset) - _hip_guide(leg, hip_offset)
    return v / np.linalg.norm(v)


def _add_visual_box(parent: ET.Element, name: str, pos, size, rgba: str, mass: float = 0.0) -> None:
    _sub(parent, "geom", name=name, type="box", pos=f(pos), size=f(size), mass=f"{mass:.6g}",
         rgba=rgba, contype="0", conaffinity="0")


def _add_visual_cylinder(parent: ET.Element, name: str, fromto, radius: float, rgba: str,
                         mass: float = 0.0) -> None:
    _sub(parent, "geom", name=name, type="cylinder", fromto=f(fromto), size=f"{radius:.6g}",
         mass=f"{mass:.6g}", rgba=rgba, contype="0", conaffinity="0")


def build_xml(cfg: DogModelConfig = DogModelConfig()) -> ET.Element:
    """Build the XML tree (without tendon rest lengths; see :func:`build_dog_xml`)."""
    root = ET.Element("mujoco", model="gradus_dog")
    _sub(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="true")
    _sub(root, "option", timestep=f"{cfg.timestep}", integrator="implicitfast", cone="elliptic", impratio="10")

    visual = _sub(root, "visual")
    _sub(visual, "global", offwidth="1280", offheight="960")
    _sub(visual, "quality", shadowsize="4096")
    _sub(visual, "scale", jointlength="0.01", jointwidth="0.002", framelength="0.02")

    asset = _sub(root, "asset")
    _sub(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512",
         rgb1="0.82 0.82 0.82", rgb2="0.72 0.72 0.72")
    _sub(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.05")
    _sub(asset, "material", name="body", rgba="0.82 0.83 0.85 1")
    _sub(asset, "material", name="leg", rgba="0.95 0.58 0.12 1")
    _sub(asset, "material", name="dark", rgba="0.16 0.17 0.19 1")
    _sub(asset, "material", name="metal", rgba="0.55 0.57 0.60 1")
    _sub(asset, "material", name="red", rgba="0.85 0.12 0.12 1")
    _sub(asset, "material", name="blue", rgba="0.12 0.35 0.85 1")

    world = _sub(root, "worldbody")
    _sub(world, "light", pos="0.6 -0.6 1.2", dir="-0.4 0.4 -1", directional="true")
    _sub(world, "light", pos="-0.6 0.6 0.8", dir="0.4 -0.4 -1", directional="true")
    _sub(world, "geom", name="floor", type="plane", size="3 3 0.1", pos="0 0 0",
         material="grid", contype="1", conaffinity="7",
         friction=f"{cfg.ground_friction:.6g} 0.005 0.0001", solref=f"{cfg.contact_timeconst:.6g} 1")

    trunk = _sub(world, "body", name="trunk", pos=f((0.0, 0.0, TRUNK_Z)))
    _sub(trunk, "freejoint", name="root")
    # Main collision box: the only body geom that can hit the ground.  Feet use
    # contype 2, so foot-body contacts are filtered by the bit masks.
    _sub(trunk, "geom", name="trunk", type="box", size="0.15 0.08 0.04", mass=f"{cfg.trunk_mass:.6g}",
         material="body", contype="4", conaffinity="1",
         friction="0.8 0.005 0.0001", solref=f"{cfg.contact_timeconst:.6g} 1")
    # Visual body shell, head, snout, ears, tail, top plate, battery, eyes, camera, nose.
    _add_visual_box(trunk, "neck_vis", (0.155, 0, 0.015), (0.025, 0.040, 0.0275), "0.82 0.83 0.85 1")
    _add_visual_box(trunk, "head_vis", (0.205, 0, 0.025), (0.0475, 0.0475, 0.0325), "0.82 0.83 0.85 1")
    _add_visual_box(trunk, "snout_vis", (0.250, 0, 0.000), (0.0275, 0.0325, 0.0175), "0.82 0.83 0.85 1")
    _add_visual_box(trunk, "earL_vis", (0.185, 0.050, 0.070), (0.0125, 0.008, 0.025), "0.95 0.58 0.12 1")
    _add_visual_box(trunk, "earR_vis", (0.185, -0.050, 0.070), (0.0125, 0.008, 0.025), "0.95 0.58 0.12 1")
    _add_visual_box(trunk, "tail_vis", (-0.170, 0, 0.020), (0.025, 0.010, 0.009), "0.95 0.58 0.12 1")
    _add_visual_box(trunk, "tailTip_vis", (-0.205, 0, 0.030), (0.015, 0.008, 0.008), "0.95 0.58 0.12 1")
    _add_visual_box(trunk, "topPlate_vis", (0, 0, 0.043), (0.110, 0.060, 0.006), "0.16 0.17 0.19 1")
    _add_visual_box(trunk, "battery_vis", (0, 0, 0.057), (0.075, 0.0475, 0.010), "0.16 0.17 0.19 1")
    for eye, y in (("eyeL", 0.025), ("eyeR", -0.025)):
        _sub(trunk, "geom", name=f"{eye}_vis", type="sphere", pos=f((0.250, y, 0.035)), size="0.008",
             mass="0", rgba="0.16 0.17 0.19 1", contype="0", conaffinity="0")
    _add_visual_cylinder(trunk, "camera_vis", (0.270, 0, 0.013, 0.285, 0, 0.013), 0.007, "0.16 0.17 0.19 1")
    _sub(trunk, "geom", name="lens_vis", type="sphere", pos="0.285 0 0.013", size="0.005", mass="0",
         rgba="0.12 0.35 0.85 1", contype="0", conaffinity="0")
    _sub(trunk, "geom", name="nose_vis", type="sphere", pos="0.282 0 -0.007", size="0.007", mass="0",
         rgba="0.16 0.17 0.19 1", contype="0", conaffinity="0")

    tendon_el = _sub(root, "tendon")
    actuator = _sub(root, "actuator")

    for leg in LEGS:
        hip_rel = _rel_hip(leg)
        knee_vec = _knee_vec(leg)
        foot_vec = _foot_vec(leg)

        # Hip fork and drum plate (visual; part of the merged shell in the CAD).
        _add_visual_box(trunk, f"{leg.name}_forkInner_vis",
                        (leg.hip[0], leg.hip[1] - leg.side * 0.0205, leg.hip[2] - 0.005 - TRUNK_Z),
                        (0.025, 0.0025, 0.025), "0.16 0.17 0.19 1")
        _add_visual_box(trunk, f"{leg.name}_forkOuter_vis",
                        (leg.hip[0], leg.hip[1] + leg.side * 0.0205, leg.hip[2] - 0.005 - TRUNK_Z),
                        (0.025, 0.0025, 0.025), "0.16 0.17 0.19 1")
        _add_visual_box(trunk, f"{leg.name}_forkTop_vis",
                        (leg.hip[0], leg.hip[1], leg.hip[2] + 0.020 - TRUNK_Z),
                        (0.025, 0.028, 0.005), "0.16 0.17 0.19 1")
        _add_visual_cylinder(trunk, f"{leg.name}_hipBolt_vis",
                             (leg.hip[0], leg.hip[1] - leg.side * 0.028, leg.hip[2] - TRUNK_Z,
                              leg.hip[0], leg.hip[1] + leg.side * 0.028, leg.hip[2] - TRUNK_Z),
                             0.005, "0.55 0.57 0.60 1")
        _add_visual_box(trunk, f"{leg.name}_drumPlate_vis",
                        (leg.hip[0], leg.hip[1] + leg.side * 0.0225, leg.hip[2] + 0.044 - TRUNK_Z),
                        (0.059, 0.0025, 0.020), "0.16 0.17 0.19 1")

        thigh = _sub(trunk, "body", name=f"{leg.name}_thigh", pos=f(hip_rel))
        _sub(thigh, "joint", name=f"{leg.name}_hip", type="hinge", axis="0 1 0",
             range=f"{-cfg.joint_range:.6g} {cfg.joint_range:.6g}",
             damping=f"{cfg.joint_damping:.6g}", armature=f"{cfg.joint_armature:.6g}",
             frictionloss=f"{cfg.joint_frictionloss:.6g}")
        _sub(thigh, "geom", name=f"{leg.name}_thigh_geom", type="capsule",
             fromto=f((0, 0, 0, *knee_vec)), size="0.010", mass=f"{cfg.thigh_mass:.6g}",
             material="leg", contype="0", conaffinity="0")
        _sub(thigh, "geom", name=f"{leg.name}_hipJoint_vis", type="sphere", pos="0 0 0", size="0.013",
             mass="0", rgba="0.16 0.17 0.19 1", contype="0", conaffinity="0")
        _sub(thigh, "geom", name=f"{leg.name}_kneeJoint_vis", type="sphere", pos=f(knee_vec), size="0.010",
             mass="0", rgba="0.16 0.17 0.19 1", contype="0", conaffinity="0")

        calf = _sub(thigh, "body", name=f"{leg.name}_calf", pos=f(knee_vec))
        _sub(calf, "joint", name=f"{leg.name}_knee", type="hinge", axis="0 1 0",
             range=f"{-cfg.joint_range:.6g} {cfg.joint_range:.6g}",
             damping=f"{cfg.joint_damping:.6g}", armature=f"{cfg.joint_armature:.6g}",
             frictionloss=f"{cfg.joint_frictionloss:.6g}")
        _sub(calf, "geom", name=f"{leg.name}_calf_geom", type="capsule",
             fromto=f((0, 0, 0, *foot_vec)), size="0.008", mass=f"{cfg.calf_mass:.6g}",
             material="leg", contype="0", conaffinity="0")
        _sub(calf, "geom", name=f"{leg.name}_foot", type="sphere", pos=f(foot_vec), size="0.015",
             mass=f"{cfg.foot_mass:.6g}", rgba="0.16 0.17 0.19 1", contype="2", conaffinity="1",
             friction=f"{cfg.ground_friction:.6g} 0.005 0.0001", solref=f"{cfg.contact_timeconst:.6g} 1",
             condim="3")

        for k, (hip_offset, knee_offset, foot_offset) in enumerate(cfg.tendon_guides):
            color = "red" if k == 0 else "blue"
            drum = _drum_center(leg, hip_offset)
            hip_guide = _hip_guide(leg, hip_offset)
            knee_guide = _knee_guide(leg, knee_offset)
            foot_anchor = _foot_anchor(leg, foot_offset)
            axis = _slide_axis(leg, hip_offset)

            # Fixed guide graphics, matching the CAD's guide bars.
            _add_visual_cylinder(trunk, f"{leg.name}_t{k}_hipGuide_vis",
                                 (hip_guide[0], leg.hip[1] - leg.side * 0.022, hip_guide[2],
                                  hip_guide[0], leg.hip[1] + leg.side * 0.022, hip_guide[2]),
                                 0.004, "0.55 0.57 0.60 1")
            _sub(trunk, "site", name=f"{leg.name}_t{k}_hipguide", pos=f(hip_guide), size="0.002",
                 rgba="0.85 0.12 0.12 1" if k == 0 else "0.12 0.35 0.85 1")
            _sub(thigh, "site", name=f"{leg.name}_t{k}_kneeguide", pos=f(knee_guide), size="0.002",
                 rgba="0.85 0.12 0.12 1" if k == 0 else "0.12 0.35 0.85 1")
            _sub(calf, "site", name=f"{leg.name}_t{k}_footanchor", pos=f(foot_anchor), size="0.002",
                 rgba="0.85 0.12 0.12 1" if k == 0 else "0.12 0.35 0.85 1")
            _sub(thigh, "geom", name=f"{leg.name}_t{k}_kneeBracket_vis", type="box",
                 pos=f((knee_vec[0] + (knee_guide[0] - knee_vec[0]) / 2, 0, knee_vec[2])),
                 size="0.009 0.005 0.005", mass="0", rgba="0.16 0.17 0.19 1",
                 contype="0", conaffinity="0")
            _sub(calf, "geom", name=f"{leg.name}_t{k}_anchor_vis", type="box", pos=f(foot_anchor),
                 size="0.006 0.006 0.006", mass="0", rgba="0.16 0.17 0.19 1", contype="0", conaffinity="0")
            # Arm that carries the anchor beside the foot (matches the CAD).
            _add_visual_cylinder(calf, f"{leg.name}_t{k}_anchorArm_vis",
                                 (*foot_vec, *foot_anchor), 0.004, "0.16 0.17 0.19 1")

            # Drum graphic and carriage proxy.
            _sub(trunk, "geom", name=f"{leg.name}_t{k}_drum_vis", type="cylinder",
                 fromto=f((drum[0], leg.hip[1] + leg.side * 0.023, drum[2],
                           drum[0], leg.hip[1] + leg.side * 0.037, drum[2])),
                 size=f"{cfg.drum_radius:.6g}", mass="0", rgba="0.85 0.12 0.12 1" if k == 0 else "0.12 0.35 0.85 1",
                 contype="0", conaffinity="0")
            # Gearbox can behind the drum (the winch reduction lives here).
            _sub(trunk, "geom", name=f"{leg.name}_t{k}_gearbox_vis", type="cylinder",
                 fromto=f((drum[0], leg.hip[1] + leg.side * 0.046, drum[2],
                           drum[0], leg.hip[1] + leg.side * 0.060, drum[2])),
                 size=f"{cfg.drum_radius * 0.95:.6g}", mass="0", rgba="0.55 0.57 0.60 1",
                 contype="0", conaffinity="0")
            carriage = _sub(trunk, "body", name=f"{leg.name}_t{k}_carriage", pos=f(drum))
            _sub(carriage, "joint", name=f"{leg.name}_t{k}_cable", type="slide", axis=f(axis),
                 range="-0.012 0.025", damping="0.01", armature=f"{cfg.reflected_inertia:.6g}")
            _sub(carriage, "geom", name=f"{leg.name}_t{k}_carriage_geom", type="box", size="0.004 0.004 0.004",
                 mass=f"{cfg.carriage_mass:.6g}", rgba="0.2 0.4 0.8 1", contype="0", conaffinity="0")
            _sub(carriage, "site", name=f"{leg.name}_t{k}_carriage", size="0.001")

            spatial = _sub(tendon_el, "spatial", name=f"{leg.name}_t{k}", width="0.0015",
                           rgba="0.85 0.12 0.12 1" if k == 0 else "0.12 0.35 0.85 1",
                           limited="true", range="0 1", solreflimit=f"{cfg.tendon_limit_timeconst:.6g} 1")
            for site in (f"{leg.name}_t{k}_carriage", f"{leg.name}_t{k}_hipguide",
                         f"{leg.name}_t{k}_kneeguide", f"{leg.name}_t{k}_footanchor"):
                _sub(spatial, "site", site=site)

            _sub(actuator, "position", name=f"{leg.name}_t{k}_servo", joint=f"{leg.name}_t{k}_cable",
                 gear=f"{1.0 / cfg.drum_radius:.6g}", kp=f"{cfg.kp:.6g}", kv=f"{cfg.kv:.6g}",
                 ctrlrange=f(cfg.angle_range), forcerange=f((-cfg.max_torque, cfg.max_torque)))

    return root


def set_rest_lengths(root: ET.Element, cfg: DogModelConfig) -> list[float]:
    """Set each tendon's upper limit to its path length at the zero pose (plus slack)."""
    # Write a temporary XML with a harmless upper limit, load it, and read the path lengths.
    tmp = ET.ElementTree(root)
    text = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(text)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lengths = [float(v) + cfg.tendon_slack for v in data.ten_length]
    for el, length in zip(root.iter("spatial"), lengths):
        el.set("range", f"0 {length:.9g}")
    return lengths


def build_dog_xml(out: str | Path = DEFAULT_OUT, cfg: DogModelConfig = DogModelConfig()) -> Path:
    """Build and write ``robot.xml``; returns its path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = build_xml(cfg)
    lengths = set_rest_lengths(root, cfg)
    ET.indent(root)
    path = out / "robot.xml"
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")

    # Reload the final file so the report reflects exactly what is on disk.
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    feet = {leg.name: float(data.geom_xpos[model.geom(f"{leg.name}_foot").id, 2]) for leg in LEGS}
    report = {
        "model": "gradus_dog",
        "source": "cad/gradus_dog.fs",
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "ntendon": model.ntendon,
        "mass_kg": float(model.body_subtreemass[model.body("trunk").id]),
        "trunk_z_m": TRUNK_Z,
        "foot_z_m": feet,
        "tendon_rest_length_m": [round(v, 6) for v in lengths],
        "drum_radius_m": cfg.drum_radius,
        "max_tension_N": cfg.max_tension,
        "drum_torque_Nm": cfg.max_torque,
        "drum_speed_rad_s": cfg.max_speed,
        "gear_ratio": cfg.gear_ratio,
        "tendon_guides_m": [list(g) for g in cfg.tendon_guides],
    }
    (out / "model_info.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(yaml.safe_dump(report, sort_keys=False))
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_dog_xml(args.out)
