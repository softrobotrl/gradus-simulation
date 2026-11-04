#!/usr/bin/env python3
"""Merge tendon routing and rod springs into an onshape-to-robot MuJoCo model.

Reads the MJCF produced by ``onshape-to-robot`` and injects:

* tendon sites at the real 2.1 mm cable holes (inner r=16 mm on every disk,
  outer r=22 mm on the hip and the upper six disks),
* eight ``<spatial>`` tendons (4 inner full-length, 4 outer upper-half),
* tension-only ``<motor>`` actuators,
* isotropic ball-joint stiffness representing the fibreglass backbone,
* mesh-derived mass (PLA density) instead of onshape-to-robot's 1e-9 kg,
* an optional fixed base and a --45 deg yaw so tendon directions are +-X/+-Y.

The hole geometry and the station transforms are read from Onshape, so there
is no hand-maintained parameter file.

Usage::

    python -m mujoco_to_mjcf.merge \
        --xml /tmp/leg/leg.xml \
        --assembly <assembly-element-id> \
        --partstudio <part-studio-element-id> \
        --out /tmp/leg/leg_merged.xml
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

from mujoco_to_mjcf.onshape import Onshape

DEFAULT_DOC = "e8da7c50681f40644f3812fb"
DEFAULT_WS = "71bd24dd8d18a5e53f7ffb25"
DEFAULT_PS = "cd41842204143b0c2cc06855"
DEFAULT_ASM = "32abf31949e6fa2193ae31aa"

INNER_R = 0.016
OUTER_R = 0.022


# --------------------------------------------------------------------------- #
# Onshape extraction
# --------------------------------------------------------------------------- #
def load_part_holes(did: str, wid: str, psid: str) -> dict:
    """Return {partId: {inner: [(x,y)...], outer: [...], zc: m}} in part-local m."""
    c = Onshape()
    st, data = c.get(f"/api/v9/partstudios/d/{did}/w/{wid}/e/{psid}/bodydetails")
    if st != 200:
        raise RuntimeError(f"bodydetails failed ({st})")

    out: dict[str, dict] = {}
    for body in data["bodies"]:
        name = (body.get("properties") or {}).get("name", "")
        if name == "PRT-BackBone":
            continue
        bore = None
        holes = []
        zmin, zmax = 1e9, -1e9
        for face in body.get("faces", []):
            s = face.get("surface", {})
            box = face.get("box")
            if box:
                zmin = min(zmin, box["minCorner"]["z"])
                zmax = max(zmax, box["maxCorner"]["z"])
            if s.get("type") != "CYLINDER":
                continue
            r = s.get("radius", 0.0)
            o = s.get("origin", {})
            if abs(r - 0.004069) < 2e-4:          # backbone bore
                bore = (o["x"], o["y"])
            elif 9e-4 < r < 1.3e-3:                # 2.1 mm cable hole
                holes.append((o["x"], o["y"]))
        if bore is None or not holes:
            continue

        inner, outer = [], []
        for hx, hy in holes:
            rel = (hx - bore[0], hy - bore[1])
            rr = math.hypot(*rel)
            if abs(rr - INNER_R) < 1e-3:
                inner.append((hx, hy))
            elif abs(rr - OUTER_R) < 1e-3:
                outer.append((hx, hy))
        out[body["id"]] = {
            "inner": inner,
            "outer": outer,
            "zc": 0.5 * (zmin + zmax),
        }
    return out


def load_stations(did: str, wid: str, eid: str, parts: dict) -> list[dict]:
    """Group top-level half-disk occurrences into stations, top to bottom."""
    c = Onshape()
    st, data = c.get(f"/api/v16/assemblies/d/{did}/w/{wid}/e/{eid}")
    if st != 200:
        raise RuntimeError(f"assembly definition failed ({st})")
    ra = data["rootAssembly"]
    inst = {i["id"]: i for i in ra["instances"]}

    rows = []
    for occ in ra["occurrences"]:
        if len(occ["path"]) != 1:
            continue
        i = inst.get(occ["path"][0])
        if not i:
            continue
        pid = i.get("partId")
        if pid not in parts:
            continue
        T = np.array(occ["transform"], dtype=float).reshape(4, 4)
        rows.append({"pid": pid, "T": T, "z": T[2, 3]})

    grouped: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        grouped.setdefault((r["pid"], round(r["z"], 4)), []).append(r)
    stations = sorted(grouped.items(), key=lambda kv: -kv[0][1])
    return [{"key": k, "instances": v} for k, v in stations]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def circular_dist(a: float, b: float) -> float:
    d = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    return d


def collect_bodies(root: ET.Element) -> list[ET.Element]:
    """Depth-first body order == MuJoCo body id order (excluding world)."""
    bodies: list[ET.Element] = []

    def walk(elem):
        for child in list(elem):
            if child.tag == "body":
                bodies.append(child)
                walk(child)

    walk(root)
    return bodies


def quat_z(angle: float) -> str:
    return f"{math.cos(angle / 2):.9g} 0 0 {math.sin(angle / 2):.9g}"


# --------------------------------------------------------------------------- #
# Main merge
# --------------------------------------------------------------------------- #
def merge(args) -> None:
    parts = load_part_holes(args.document, args.workspace, args.partstudio)
    stations = load_stations(args.document, args.workspace, args.assembly, parts)
    print(f"* {len(parts)} parts, {len(stations)} stations")

    # --- body world poses at q=0 ------------------------------------------ #
    import mujoco

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tree = ET.parse(args.xml)
    mujoco_root = tree.getroot()
    worldbody = mujoco_root.find("worldbody")
    bodies = collect_bodies(worldbody)
    if len(bodies) != len(stations):
        raise RuntimeError(f"{len(bodies)} bodies vs {len(stations)} stations")

    # --- reference angles from the first station --------------------------- #
    def holes_world(station):
        out = []
        for r in station["instances"]:
            pid = r["pid"]
            zc = parts[pid]["zc"]
            for kind in ("inner", "outer"):
                for (x, y) in parts[pid][kind]:
                    p = (r["T"] @ np.array([x, y, zc, 1.0]))[:3]
                    ang = math.atan2(p[1], p[0])
                    out.append((kind, ang, p))
        return out

    ref = holes_world(stations[0])
    ref_inner = sorted(a for k, a, _ in ref if k == "inner")
    # rotate so the hole nearest +45 deg (the CAD's natural diagonal) is +X
    front = min(ref_inner, key=lambda a: circular_dist(a, math.radians(45)))
    yaw = -front
    dir_names = ["front", "left", "back", "right"]

    def dir_of(angle: float) -> str:
        b = (angle + yaw) % (2 * math.pi)
        idx = int(round(b / (math.pi / 2))) % 4
        return dir_names[idx]

    # --- sites ------------------------------------------------------------- #
    site_names: dict[tuple, str] = {}
    site_world: dict[tuple, np.ndarray] = {}
    for i, station in enumerate(stations):
        body = bodies[i]
        xpos = data.xpos[i + 1].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.xquat[i + 1])
        R = R.reshape(3, 3)
        holes = holes_world(station)

        seen = set()
        for kind, ang, p in holes:
            name_dir = dir_of(ang)
            key = (kind, name_dir)
            if key in seen:
                continue
            seen.add(key)
            local = R.T @ (p - xpos)
            sname = f"tendon_{kind}_{name_dir}_{i:02d}"
            ET.SubElement(body, "site", {
                "name": sname,
                "pos": " ".join(f"{v:.9g}" for v in local),
                "size": "0.0015",
                "rgba": "0.9 0.1 0.1 1" if kind == "inner" else "0.1 0.4 0.9 1",
            })
            site_names[(i, kind, name_dir)] = sname
            site_world[(i, kind, name_dir)] = p

    # --- tendons ----------------------------------------------------------- #
    tendon = ET.SubElement(mujoco_root, "tendon")
    tendon_specs = []

    def polyline_len(keys: list[tuple]) -> float:
        return float(sum(
            np.linalg.norm(site_world[keys[j + 1]] - site_world[keys[j]])
            for j in range(len(keys) - 1)
        ))

    for d in dir_names:
        keys = [(i, "inner", d) for i in range(len(stations))]
        tendon_specs.append((f"tendon_inner_{d}", [site_names[k] for k in keys], polyline_len(keys)))
    for d in dir_names:
        keys = [(i, "outer", d) for i in range(len(stations)) if (i, "outer", d) in site_names]
        if len(keys) >= 2:
            tendon_specs.append((f"tendon_outer_{d}", [site_names[k] for k in keys], polyline_len(keys)))

    for name, sites, _ in tendon_specs:
        sp = ET.SubElement(tendon, "spatial", {"name": name, "width": "0.001"})
        for s in sites:
            ET.SubElement(sp, "site", {"site": s})
        print(f"* {name}: {len(sites)} sites")

    # --- actuators --------------------------------------------------------- #
    actuator = ET.SubElement(mujoco_root, "actuator")
    if args.servo:
        fmax = args.servo_torque / args.spool_radius
        stroke = args.spool_radius * args.servo_travel
        print(f"* servo: torque={args.servo_torque:.2f}Nm spool={args.spool_radius*1000:.1f}mm "
              f"-> Fmax={fmax:.1f}N stroke={stroke*1000:.1f}mm travel={math.degrees(args.servo_travel):.0f}deg")
        for name, _, l0 in tendon_specs:
            ET.SubElement(actuator, "position", {
                "name": "servo_" + name,
                "tendon": name,
                "kp": f"{args.servo_kp:.6g}",
                "kv": f"{args.servo_kv:.6g}",
                "ctrlrange": f"{l0 - stroke:.6g} {l0:.6g}",
                "forcerange": f"{-fmax:.6g} 0",
            })
    else:
        for name, _, _ in tendon_specs:
            ET.SubElement(actuator, "motor", {
                "name": "act_" + name,
                "tendon": name,
                "ctrlrange": f"0 {args.tendon_fmax}",
                "gear": "-1",
            })

    # --- rod spring on every ball joint ------------------------------------ #
    cz = [s["key"][1] + parts[s["key"][0]]["zc"] for s in stations]
    pitch = float(np.mean(np.abs(np.diff(cz))))
    I = math.pi * args.rod_d ** 4 / 64.0
    k = args.rod_E * I / pitch
    print(f"* rod: d={args.rod_d*1000:.2f}mm E={args.rod_E/1e9:.1f}GPa L={pitch*1000:.2f}mm -> k={k:.1f} N.m/rad per joint")
    for j in mujoco_root.iter("joint"):
        if j.get("type") == "ball":
            j.set("stiffness", f"{k:.6g}")
            j.set("damping", "0.5")
            j.set("armature", "0.01")

    # --- mass from PLA density --------------------------------------------- #
    for default in mujoco_root.iter("default"):
        cls = default.get("class")
        for geom in default.findall("geom"):
            if cls == "visual":
                geom.set("density", "0")
            elif cls == "collision":
                geom.set("density", str(args.density))
    for inertial in list(mujoco_root.iter("inertial")):
        parent = None
        for p in mujoco_root.iter():
            if inertial in list(p):
                parent = p
                break
        if parent is not None:
            parent.remove(inertial)

    # --- base handling ----------------------------------------------------- #
    if args.fix_base:
        for fj in list(mujoco_root.iter("freejoint")):
            for p in mujoco_root.iter():
                if fj in list(p):
                    p.remove(fj)
                    break

    # --- yaw so tendon directions are +-X/+-Y ------------------------------ #
    if args.rotate:
        # root body may already carry a quat (hip) -- compose with the yaw
        root_body = bodies[0]
        q = root_body.get("quat", "1 0 0 0").split()
        q = np.array([float(x) for x in q])
        yq = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = yq
        qq = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
        qq /= np.linalg.norm(qq)
        root_body.set("quat", " ".join(f"{v:.9g}" for v in qq))
        print(f"* yaw {math.degrees(yaw):.2f} deg")

    ET.indent(tree, space="  ")
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"* wrote {args.out}")

    # --- validation -------------------------------------------------------- #
    m = mujoco.MjModel.from_xml_path(args.out)
    total_mass = sum(float(m.body_mass[i]) for i in range(m.nbody))
    print(f"* check: bodies={m.nbody} joints={m.njnt} sites={m.nsite} "
          f"tendons={m.ntendon} actuators={m.nu} mass={total_mass*1000:.1f} g")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", required=True, help="onshape-to-robot MJCF")
    p.add_argument("--out", required=True, help="merged MJCF output")
    p.add_argument("--document", default=DEFAULT_DOC)
    p.add_argument("--workspace", default=DEFAULT_WS)
    p.add_argument("--assembly", default=DEFAULT_ASM)
    p.add_argument("--partstudio", default=DEFAULT_PS)
    p.add_argument("--rod-d", type=float, default=0.0079375, help="rod diameter (m); 5/16 in")
    p.add_argument("--rod-E", type=float, default=30e9, help="rod Young's modulus (Pa)")
    p.add_argument("--density", type=float, default=1240.0, help="PLA density kg/m^3")
    p.add_argument("--tendon-fmax", type=float, default=2000.0)
    p.add_argument("--servo", action="store_true",
                   help="model servos (position on tendon length) instead of force motors")
    p.add_argument("--servo-torque", type=float, default=2.0, help="servo stall torque (N.m)")
    p.add_argument("--spool-radius", type=float, default=0.010, help="tendon spool radius (m)")
    p.add_argument("--servo-travel", type=float, default=math.pi, help="max spool travel (rad)")
    p.add_argument("--servo-kp", type=float, default=5000.0, help="servo position gain")
    p.add_argument("--servo-kv", type=float, default=50.0, help="servo velocity gain")
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    p.add_argument("--free-base", dest="fix_base", action="store_false")
    p.set_defaults(rotate=True, fix_base=True)
    args = p.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
