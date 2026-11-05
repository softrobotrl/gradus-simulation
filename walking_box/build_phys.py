#!/usr/bin/env python3
"""Make a fast physics-only quadruped XML for MJX.

The full ``quadruped.xml`` keeps all the visual/collision meshes.  MJX compiles
a broadphase over every geom, which is far too slow here.  This script:

* copies the full body mass/inertia into explicit ``<inertial>`` elements,
* removes all mesh geoms,
* keeps only the floor, trunk box and four foot spheres (all massless),
* drops the unused mesh assets.

The full model is still used for rendering; the physics-only model has the same
bodies, joints, sites, tendons, actuators and mass distribution.

Usage::

    python -m walking_box.build_phys
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "urdf/gradus/walk/quadruped.xml"
OUT = ROOT / "urdf/gradus/walk/quadruped_phys.xml"

KEEP_GEOMS = {"floor", "trunk"}


def build() -> None:
    full = mujoco.MjModel.from_xml_path(str(SRC))
    tree = ET.parse(SRC)
    root = tree.getroot()

    # explicit inertials for every body
    for body in root.iter("body"):
        name = body.get("name")
        if not name:
            continue
        bid = full.body(name).id
        inertial = ET.Element("inertial", {
            "pos": " ".join(f"{v:.9g}" for v in full.body_ipos[bid]),
            "quat": " ".join(f"{v:.9g}" for v in full.body_iquat[bid]),
            "mass": f"{full.body_mass[bid]:.9g}",
            "diaginertia": " ".join(f"{v:.9g}" for v in full.body_inertia[bid]),
        })
        body.insert(0, inertial)

    # keep only floor/trunk/foot geoms; make them massless
    for parent in list(root.iter()):
        for geom in list(parent.findall("geom")):
            name = geom.get("name", "")
            if name in KEEP_GEOMS or name.endswith("_foot"):
                geom.set("mass", "0")
            else:
                parent.remove(geom)

    # meshes/materials are no longer referenced
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            asset.remove(child)

    # fixed contact/constraint capacity for MuJoCo Warp
    option = root.find("option")
    size = ET.Element("size", {"nconmax": "2048", "njmax": "512"})
    if option is not None:
        root.insert(list(root).index(option) + 1, size)
    else:
        root.insert(0, size)

    ET.indent(root, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=True)
    print(f"* wrote {OUT}")

    m = mujoco.MjModel.from_xml_path(str(OUT))
    d = mujoco.MjData(m)
    d.ctrl[:] = m.actuator_ctrlrange[:, 1]
    mujoco.mj_forward(m, d)
    for _ in range(300):
        mujoco.mj_step(m, d)
    mass = float(sum(m.body_mass))
    print(f"* bodies={m.nbody} geoms={m.ngeom} nq={m.nq} nv={m.nv} "
          f"tendons={m.ntendon} actuators={m.nu} mass={mass:.2f} kg")
    print(f"* trunk z after 300 steps: {d.qpos[2]:.4f}, contacts={d.ncon}")


if __name__ == "__main__":
    build()
