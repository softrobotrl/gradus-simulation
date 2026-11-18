#!/usr/bin/env python3
"""Mount the demo leg on a one-dimensional sliding rig.

Reads ``urdf/gradus/demo/leg.xml`` (hip fixed at the origin, foot pointing
down, 8 tendon servos) and writes ``urdf/gradus/demo/rig.xml``:

* the hip is rigidly bolted to a carriage,
* the carriage has a single prismatic joint along +X (a linear rail),
* the only contact geom is a small sphere at the ``tip`` site,
* the leg mesh collisions are disabled, so the foot is the only contact.

The rail height is derived from the actual generated leg: the hip sits a few
millimetres lower than the straight-leg length, so the leg starts slightly
compressed (and the environment adds a small intentional bend at reset).

Usage::

    python -m onshape_to_mjcf.build --out urdf/gradus/demo
    python -m single_leg_demo.build_rig
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
LEG_XML = ROOT / "urdf/gradus/demo/leg.xml"
OUT_XML = ROOT / "urdf/gradus/demo/rig.xml"

FOOT_RADIUS = 0.015
FOOT_MASS = 0.02
SLIDER_MASS = 2.0
RAIL_HALF = 0.45
RAIL_COMPRESSION = 0.002      # hip height below the straight-leg contact height
SLIDER_DAMPING = 0.6
SLIDER_FRICTION = 0.2


def _copy_defaults(root_in: ET.Element, root_out: ET.Element) -> None:
    for tag in ("compiler", "default", "asset"):
        elem = root_in.find(tag)
        if elem is not None:
            root_out.append(copy.deepcopy(elem))
    compiler = root_out.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "assets")


def build() -> None:
    if not LEG_XML.exists():
        raise SystemExit(
            f"{LEG_XML} is missing; run `python -m onshape_to_mjcf.build --out urdf/gradus/demo` first"
        )

    # --- measure the generated leg ---------------------------------------- #
    leg_model = mujoco.MjModel.from_xml_path(str(LEG_XML))
    leg_data = mujoco.MjData(leg_model)
    mujoco.mj_forward(leg_model, leg_data)
    tip_z = float(leg_data.site("tip").xpos[2])
    leg_len = -tip_z + FOOT_RADIUS
    rail_z = leg_len - RAIL_COMPRESSION
    print(f"* leg length hip->foot bottom = {leg_len:.4f} m, rail z = {rail_z:.4f} m")

    # --- build the rig XML -------------------------------------------------- #
    leg_root = ET.parse(LEG_XML).getroot()
    leg_world = leg_root.find("worldbody")
    leg_bodies = [b for b in leg_world.findall("body") if b.get("name")]
    if len(leg_bodies) != 1:
        raise RuntimeError(f"expected one leg root body, found {len(leg_bodies)}")
    hip = leg_bodies[0]

    leg_tendon = next((t for t in leg_root.findall("tendon") if len(t)), None)
    leg_actuator = next((a for a in leg_root.findall("actuator") if len(a)), None)
    if leg_tendon is None or leg_actuator is None:
        raise RuntimeError("leg.xml has no tendon/actuator elements")

    root = ET.Element("mujoco", {"model": "single_leg_demo"})
    _copy_defaults(leg_root, root)

    ET.SubElement(root, "option", {
        "timestep": "0.002",
        "cone": "elliptic",
        "impratio": "1",
    })

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {
        "offwidth": "960", "offheight": "720", "azimuth": "135", "elevation": "-18",
    })
    ET.SubElement(visual, "headlight", {
        "diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0",
    })

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "ground", "builtin": "checker", "mark": "edge",
        "rgb1": "0.22 0.30 0.36", "rgb2": "0.12 0.18 0.24",
        "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "ground", "texture": "ground", "texuniform": "true",
        "texrepeat": "8 8", "reflectance": "0.15",
    })

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 2.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05", "pos": "0 0 0",
        "contype": "1", "conaffinity": "1", "friction": "1.0 0.01 0.001",
        "material": "ground",
    })

    # static rail (visual only; the carriage joint carries the constraint)
    ET.SubElement(wb, "geom", {
        "name": "rail", "type": "box",
        "size": f"{RAIL_HALF + 0.06:.4f} 0.012 0.012",
        "pos": f"0 0 {rail_z + 0.028:.6f}",
        "contype": "0", "conaffinity": "0",
        "rgba": "0.35 0.35 0.38 1",
    })

    slider = ET.SubElement(wb, "body", {
        "name": "slider", "pos": f"0 0 {rail_z:.6f}",
    })
    ET.SubElement(slider, "joint", {
        "name": "slide", "type": "slide", "axis": "1 0 0",
        "range": f"{-RAIL_HALF:.4f} {RAIL_HALF:.4f}", "limited": "true",
        "damping": str(SLIDER_DAMPING), "frictionloss": str(SLIDER_FRICTION),
        "armature": "0.001",
    })
    ET.SubElement(slider, "geom", {
        "name": "carriage", "type": "box", "size": "0.055 0.040 0.022",
        "mass": str(SLIDER_MASS), "contype": "0", "conaffinity": "0",
        "rgba": "0.55 0.55 0.60 1",
    })
    slider.append(copy.deepcopy(hip))

    # --- foot sphere and collision filtering ------------------------------- #
    foot_site = None
    for body in slider.iter("body"):
        site = body.find("site[@name='tip']")
        if site is not None:
            foot_site = (body, site)
            break
    if foot_site is None:
        raise RuntimeError("no tip site in the leg model")
    foot_body, tip_site = foot_site
    ET.SubElement(foot_body, "geom", {
        "name": "foot", "type": "sphere", "size": str(FOOT_RADIUS),
        "pos": tip_site.get("pos", "0 0 0"), "mass": str(FOOT_MASS),
        "contype": "1", "conaffinity": "1",
        "friction": "1.0 0.01 0.001", "rgba": "0.2 0.8 0.2 1",
    })
    for geom in slider.iter("geom"):
        if geom.get("class") == "collision":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    # --- tendons and actuators --------------------------------------------- #
    root.append(copy.deepcopy(leg_tendon))
    root.append(copy.deepcopy(leg_actuator))

    ET.indent(root, space="  ")
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(OUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"* wrote {OUT_XML}")

    # --- validate ----------------------------------------------------------- #
    m = mujoco.MjModel.from_xml_path(str(OUT_XML))
    d = mujoco.MjData(m)
    d.ctrl[:] = m.actuator_ctrlrange[:, 1]
    mujoco.mj_forward(m, d)
    for _ in range(400):
        mujoco.mj_step(m, d)
    tip = d.site("tip").xpos
    print(f"* bodies={m.nbody} joints={m.njnt} nq={m.nq} nv={m.nv} "
          f"tendons={m.ntendon} actuators={m.nu} mass={m.body_subtreemass[1]:.2f} kg")
    print(f"* settled tip z={tip[2]:.4f} m  slider x={d.qpos[0]:.4f} m  contacts={d.ncon}")
    if m.nu != 8 or m.ntendon != 8:
        raise RuntimeError("expected 8 tendons and 8 actuators")


if __name__ == "__main__":
    build()
