#!/usr/bin/env python3
"""Strap four copies of the generated tendon leg onto a simple box trunk.

Reads ``urdf/gradus/walk/leg.xml`` (tip-down, with a ``tip`` site) and writes
``urdf/gradus/walk/quadruped.xml``.

Each leg keeps its eight tendon servos (32 actuators). Leg mesh collisions are
disabled; the trunk box, four foot spheres and the floor carry the contact.
That keeps MJX fast enough for PPO.

Usage::

    python -m walking_box.build_quadruped
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
LEG_XML = ROOT / "urdf/gradus/walk/leg.xml"
OUT_XML = ROOT / "urdf/gradus/walk/quadruped.xml"

# prefix, trunk-local hip position
LEGS = [
    ("fl", (0.14, 0.09, -0.04)),
    ("fr", (0.14, -0.09, -0.04)),
    ("bl", (-0.14, 0.09, -0.04)),
    ("br", (-0.14, -0.09, -0.04)),
]

TRUNK_Z = 0.6525
TRUNK_SIZE = (0.18, 0.11, 0.04)
TRUNK_MASS = 1.5
FOOT_RADIUS = 0.015


def prefix_names(elem: ET.Element, prefix: str) -> None:
    for e in elem.iter():
        if "name" in e.attrib:
            e.set("name", f"{prefix}_{e.get('name')}")


def add_foot(body: ET.Element, prefix: str) -> None:
    for part in body.iter("body"):
        for site in part.findall("site"):
            if site.get("name", "").endswith("_tip"):
                ET.SubElement(part, "geom", {
                    "name": f"{prefix}_foot",
                    "type": "sphere",
                    "size": str(FOOT_RADIUS),
                    "pos": site.get("pos", "0 0 0"),
                    "mass": "0.02",
                    "contype": "1",
                    "conaffinity": "1",
                    "friction": "1.0 0.01 0.001",
                    "rgba": "0.2 0.8 0.2 1",
                })
                return
    raise RuntimeError(f"no tip site in {prefix} foot")


def disable_leg_collisions(body: ET.Element) -> None:
    for geom in body.iter("geom"):
        if geom.get("class") == "collision":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def build() -> None:
    leg_root = ET.parse(LEG_XML).getroot()
    worldbody = leg_root.find("worldbody")
    leg_body = worldbody.find("body") if worldbody is not None else None
    # onshape-to-robot emits an empty <actuator/> and merge.py appends a second
    # one; pick the one that actually has entries.
    leg_tendon = next((t for t in leg_root.findall("tendon") if len(t)), None)
    leg_actuator = next((a for a in leg_root.findall("actuator") if len(a)), None)
    if leg_body is None or leg_tendon is None or leg_actuator is None:
        raise RuntimeError("unexpected leg XML layout")

    root = ET.Element("mujoco", {"model": "walking_box"})
    compiler = leg_root.find("compiler")
    if compiler is not None:
        compiler = copy.deepcopy(compiler)
        compiler.set("meshdir", "assets")
        root.append(compiler)

    ET.SubElement(root, "option", {
        "timestep": "0.002", "cone": "elliptic", "impratio": "1",
    })

    for tag in ("default", "asset"):
        elem = leg_root.find(tag)
        if elem is not None:
            root.append(copy.deepcopy(elem))

    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", {
        "diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0",
    })
    ET.SubElement(vis, "global", {"azimuth": "140", "elevation": "-20"})

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 2", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05", "pos": "0 0 0",
        "contype": "1", "conaffinity": "1", "friction": "1.0 0.01 0.001",
        "rgba": "0.45 0.45 0.45 1",
    })

    trunk = ET.SubElement(wb, "body", {"name": "trunk", "pos": f"0 0 {TRUNK_Z}"})
    ET.SubElement(trunk, "freejoint", {"name": "trunk_free"})
    ET.SubElement(trunk, "geom", {
        "name": "trunk", "type": "box",
        "size": f"{TRUNK_SIZE[0]} {TRUNK_SIZE[1]} {TRUNK_SIZE[2]}",
        "mass": str(TRUNK_MASS), "contype": "1", "conaffinity": "1",
        "friction": "1.0 0.01 0.001", "rgba": "0.72 0.52 0.28 1",
    })

    for prefix, pos in LEGS:
        leg = copy.deepcopy(leg_body)
        prefix_names(leg, prefix)
        leg.set("pos", " ".join(f"{v:.9g}" for v in pos))
        disable_leg_collisions(leg)
        add_foot(leg, prefix)
        trunk.append(leg)

    tendon = ET.SubElement(root, "tendon")
    for prefix, _ in LEGS:
        for sp in leg_tendon.findall("spatial"):
            new = copy.deepcopy(sp)
            new.set("name", f"{prefix}_{new.get('name')}")
            for site in new.findall("site"):
                site.set("site", f"{prefix}_{site.get('site')}")
            tendon.append(new)

    actuator = ET.SubElement(root, "actuator")
    for prefix, _ in LEGS:
        for act in list(leg_actuator):
            new = copy.deepcopy(act)
            new.set("name", f"{prefix}_{new.get('name')}")
            new.set("tendon", f"{prefix}_{new.get('tendon')}")
            actuator.append(new)

    ET.indent(root, space="  ")
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(OUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"* wrote {OUT_XML}")

    m = mujoco.MjModel.from_xml_path(str(OUT_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tips = [m.site(f"{p}_tip").id for p, _ in LEGS]
    mass = float(sum(m.body_mass))
    print(f"* bodies={m.nbody} joints={m.njnt} nq={m.nq} nv={m.nv} "
          f"sites={m.nsite} tendons={m.ntendon} actuators={m.nu} mass={mass:.2f} kg")
    print("* foot z:", [round(float(d.site_xpos[t, 2]), 4) for t in tips])
    print("* trunk z:", round(float(d.qpos[2]), 4))
    if m.nu != 32 or m.ntendon != 32:
        raise RuntimeError("expected 32 tendons and 32 actuators")


if __name__ == "__main__":
    build()
