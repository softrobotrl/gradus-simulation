"""Write a synthetic leg.xml in the format mujoco_to_mjcf merges, for work without Onshape access.

The structure and numbers come from mujoco_to_mjcf/PIPELINE.md: 14 stations, 13 stiff ball joints at the mid-gaps,
4 inner tendons over all stations, 4 outer tendons over the first 7, position servos on tendon length. It is a stand-in
with plain disks: results on it say nothing about the real CAD.

Usage: python -m genesis_leg.leg_fixture --out pipeline/build/genesis/fixture_leg [--tip-up] [--stl disk.stl]
"""

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

BUILD_ROOT = Path(__file__).resolve().parents[1] / "pipeline/build/genesis"
PITCH = 0.04464     # m between stations
JUNCTION = 0.015    # m between the two leg segments
N_STATIONS = 14
INNER_R, OUTER_R = 0.016, 0.022
OUTER_STATIONS = 7  # hip + the first segment
DIRECTIONS = {"front": (1, 0), "left": (0, 1), "back": (-1, 0), "right": (0, -1)}
DISK_RADIUS, DISK_THICKNESS = 0.03, 0.012
TOTAL_MASS = 1.24   # kg
ROD_DIAMETER, ROD_E = 0.0079375, 30e9
SERVO_TORQUE, SPOOL_RADIUS, SERVO_TRAVEL = 2.0, 0.01, math.pi
SERVO_KP, SERVO_KV = 5000.0, 50.0


def _vector(*values):
    return " ".join(f"{value:.9g}" for value in values)


def station_names():
    return ["hip"] + [f"S1_{i}" for i in range(1, 7)] + [f"S2_{i}" for i in range(1, 7)] + ["foot"]


def write_fixture(out, tip_up=False, stl=None):
    """Write `out`/leg.xml and its mesh. stl: one disk (metres, centred, axis along z) to use instead of the plain one."""
    out = Path(out)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    disk = out / "assets/disk.stl"
    if stl:
        shutil.copyfile(stl, disk)
    else:
        trimesh.creation.cylinder(radius=DISK_RADIUS, height=DISK_THICKNESS, sections=48).export(disk)
    volume = abs(trimesh.load(disk).volume)

    gaps = [JUNCTION if i == 6 else PITCH for i in range(N_STATIONS - 1)]
    stiffness = ROD_E * math.pi * ROD_DIAMETER**4 / 64 / float(np.mean(gaps))
    fmax, stroke = SERVO_TORQUE / SPOOL_RADIUS, SPOOL_RADIUS * SERVO_TRAVEL

    root = ET.Element("mujoco", model="leg")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets", autolimits="true")
    default = ET.SubElement(root, "default")
    visual = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(visual, "geom", type="mesh", contype="0", conaffinity="0", group="2", density="0")
    collision = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", type="mesh", group="3", density=f"{TOTAL_MASS / N_STATIONS / volume:.9g}")
    ET.SubElement(ET.SubElement(root, "asset"), "mesh", name="disk", file="disk.stl")

    parent = ET.SubElement(root, "worldbody")
    for i, name in enumerate(station_names()):
        pos = (0, 0, 0) if i == 0 else (0, 0, -gaps[i - 1])
        body = ET.SubElement(parent, "body", name=name, pos=_vector(*pos))
        if i == 0 and tip_up:
            body.set("quat", "0 1 0 0")  # 180 degrees about x, as mujoco_to_mjcf.build --tip-up
        if i > 0:
            ET.SubElement(body, "joint", name=f"{i - 1:02d}", type="ball", pos=_vector(0, 0, gaps[i - 1] / 2),
                          stiffness=f"{stiffness:.6g}", damping="0.5", armature="0.01")
        ET.SubElement(body, "geom", {"class": "visual", "mesh": "disk"})
        ET.SubElement(body, "geom", {"class": "collision", "mesh": "disk"})
        for direction, (x, y) in DIRECTIONS.items():
            ET.SubElement(body, "site", name=f"tendon_inner_{direction}_{i:02d}",
                          pos=_vector(INNER_R * x, INNER_R * y, 0), size="0.0015")
            if i < OUTER_STATIONS:
                ET.SubElement(body, "site", name=f"tendon_outer_{direction}_{i:02d}",
                              pos=_vector(OUTER_R * x, OUTER_R * y, 0), size="0.0015")
        parent = body

    tendon = ET.SubElement(root, "tendon")
    actuator = ET.SubElement(root, "actuator")
    # Straight tendons in the straight pose, so rest length is the sum of the gaps they span.
    for kind, stations in (("inner", N_STATIONS), ("outer", OUTER_STATIONS)):
        rest = sum(gaps[:stations - 1])
        for direction in DIRECTIONS:
            name = f"tendon_{kind}_{direction}"
            spatial = ET.SubElement(tendon, "spatial", name=name, width="0.001")
            for i in range(stations):
                ET.SubElement(spatial, "site", site=f"{name}_{i:02d}")
            ET.SubElement(actuator, "position", name=f"servo_{name}", tendon=name, kp=f"{SERVO_KP:.6g}",
                          kv=f"{SERVO_KV:.6g}", ctrlrange=_vector(rest - stroke, rest), forcerange=_vector(-fmax, 0))

    ET.indent(root)
    xml = out / "leg.xml"
    xml.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return xml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BUILD_ROOT / "fixture_leg")
    parser.add_argument("--tip-up", action="store_true")
    parser.add_argument("--stl", type=Path, help="one disk STL to use instead of the plain cylinder")
    args = parser.parse_args()
    print(f"wrote {write_fixture(args.out, args.tip_up, args.stl)}")
