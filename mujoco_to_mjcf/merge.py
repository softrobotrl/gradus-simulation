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

``--nested`` handles the newer ``Sim-Leg-Assembly`` demo, whose 14 stations are
sub-assemblies and whose top-level mates are not named ``dof_*``.  In that mode
``onshape-to-robot`` writes 14 free bodies, so this script rebuilds the chain
from the nested occurrence transforms before adding the same sites/tendons.

Usage::

    python -m mujoco_to_mjcf.merge \
        --xml /tmp/leg/leg.xml \
        --assembly <assembly-element-id> \
        --partstudio <part-studio-element-id> \
        --out /tmp/leg/leg_merged.xml

    python -m mujoco_to_mjcf.merge --nested \
        --xml /tmp/demo/leg_base.xml \
        --assembly <Sim-Leg-Assembly-id> \
        --out urdf/gradus/demo/leg.xml
"""

from __future__ import annotations

import argparse
import math
import re
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
BORE_R = 0.004069

DIR_NAMES = ["front", "left", "back", "right"]


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
# Nested assembly extraction (Sim-Leg-Assembly)
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    """Match onshape-to-robot's body names, e.g. ``Shin V4 <2>`` -> ``shin_v4_2``."""
    m = re.search(r"<\s*(\d+)\s*>", name)
    base = name[: m.start()] if m else name
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    if m and m.group(1) != "1":
        slug = f"{slug}_{m.group(1)}"
    return slug


def _main_workspace(c: Onshape, did: str) -> str:
    st, data = c.get(f"/api/v6/documents/d/{did}/workspaces")
    if st != 200 or not isinstance(data, list) or not data:
        raise RuntimeError(f"workspaces failed for {did} ({st})")
    for ws in data:
        if ws.get("name") == "Main":
            return ws["id"]
    return data[0]["id"]


def _part_geometry(c: Onshape, did: str, wid: str, psid: str,
                   bore_r: float, inner_r: float, outer_r: float) -> list[dict]:
    """Bore centre, tendon holes and z extent for each body in a part studio.

    Guide holes are identified by their radial distance from the backbone bore
    instead of their diameter, so the M4 screw holes and their counterbores are
    ignored.  Both the old (16/22 mm, 2.1 mm holes) and the demo
    (19.1/29.1 mm, 6.25 mm holes) layouts work.
    """
    st, data = c.get(f"/api/v9/partstudios/d/{did}/w/{wid}/e/{psid}/bodydetails")
    if st != 200:
        raise RuntimeError(f"bodydetails failed ({st})")
    out = []
    for body in data["bodies"]:
        name = (body.get("properties") or {}).get("name", "")
        if name == "PRT-BackBone":
            continue
        bore = None
        cand = []
        zmin, zmax = 1e9, -1e9
        for face in body.get("faces", []):
            box = face.get("box")
            if box:
                zmin = min(zmin, box["minCorner"]["z"])
                zmax = max(zmax, box["maxCorner"]["z"])
            s = face.get("surface", {})
            if s.get("type") != "CYLINDER":
                continue
            r = s.get("radius", 0.0)
            o = s.get("origin", {})
            # The demo disk has 5 mm lightening holes only 0.14 mm larger than
            # the backbone bore, so this tolerance has to stay tight.
            if abs(r - bore_r) < 1e-4:
                bore = (o["x"], o["y"])
            elif r < 5.0e-3:
                cand.append((o["x"], o["y"]))
        if bore is None:
            continue
        holes = []
        for hx, hy in cand:
            rr = math.hypot(hx - bore[0], hy - bore[1])
            if abs(rr - inner_r) < 6e-4:
                holes.append(("inner", hx, hy))
            elif abs(rr - outer_r) < 6e-4:
                holes.append(("outer", hx, hy))
        if not holes:
            continue
        out.append({
            "bore": bore,
            "holes": holes,
            "zmin": zmin,
            "zmax": zmax,
            "zc": 0.5 * (zmin + zmax),
        })
    if not out:
        raise RuntimeError(f"no tendon disks found in part studio {psid}")
    return out


def load_nested_stations(did: str, wid: str, eid: str, bore_r: float,
                         inner_r: float, outer_r: float) -> list[dict]:
    """Return 14 stations (top to bottom) from a nested leg assembly.

    Each top-level instance is a sub-assembly holding the two printed halves of
    one disk.  The nested occurrence transforms returned by Onshape are already
    in the root-assembly frame, so no manual transform composition is needed.
    """
    c = Onshape()
    st, data = c.get(f"/api/v16/assemblies/d/{did}/w/{wid}/e/{eid}")
    if st != 200:
        raise RuntimeError(f"assembly definition failed ({st})")
    ra = data["rootAssembly"]
    top = {i["id"]: i for i in ra["instances"]}

    ws_cache: dict[str, str] = {}

    def workspace(doc: str) -> str:
        if doc not in ws_cache:
            ws_cache[doc] = _main_workspace(c, doc)
        return ws_cache[doc]

    # Map each sub-assembly's internal occurrence ids to their part studios.
    sub: dict[str, dict] = {}
    sub_cache: dict[tuple, dict] = {}
    for inst in top.values():
        if inst.get("type") != "Assembly":
            continue
        dd, ed = inst["documentId"], inst["elementId"]
        key = (dd, ed)
        if key not in sub_cache:
            ww = workspace(dd)
            st2, a = c.get(f"/api/v16/assemblies/d/{dd}/w/{ww}/e/{ed}")
            if st2 != 200:
                raise RuntimeError(f"sub-assembly {ed} failed ({st2})")
            sub_cache[key] = {x["id"]: x for x in a["rootAssembly"]["instances"]}
        sub[inst["id"]] = sub_cache[key]

    cache: dict[tuple, list[dict]] = {}

    def geom_of(dd: str, ww: str, ps: str) -> list[dict]:
        key = (dd, ww, ps)
        if key not in cache:
            cache[key] = _part_geometry(c, dd, ww, ps, bore_r, inner_r, outer_r)
        return cache[key]

    stations: "OrderedDict[str, dict]" = OrderedDict()
    for occ in ra["occurrences"]:
        if len(occ["path"]) != 2:
            continue
        ti, ii = occ["path"]
        inst = top.get(ti)
        inner = sub.get(ti, {}).get(ii)
        if inst is None or inner is None:
            continue
        dd = inst["documentId"]
        ww = workspace(dd)
        ps = inner["elementId"]
        T = np.array(occ["transform"], dtype=float).reshape(4, 4)
        site = stations.setdefault(ti, {
            "name": inst.get("name", ""),
            "body": _slug(inst.get("name", "")),
            "centers": [],
            "parts": [],
            "holes": [],
            "zmin": 1e9,
        })
        for g in geom_of(dd, ww, ps):
            ctr = (T @ np.array([g["bore"][0], g["bore"][1], g["zc"], 1.0]))[:3]
            site["centers"].append(ctr)
            site["parts"].append((T, g))
            # every part is only rotated about Z, so zmin is unambiguous
            site["zmin"] = min(site["zmin"], T[2, 3] + g["zmin"])
            for kind, hx, hy in g["holes"]:
                p = (T @ np.array([hx, hy, g["zc"], 1.0]))[:3]
                site["holes"].append((kind, p))

    out = []
    for s in stations.values():
        s["center"] = np.mean(s["centers"], axis=0)
        out.append(s)
    out.sort(key=lambda s: -s["center"][2])
    print(f"* nested: {len(out)} stations")
    return out


def _vec(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def _dir_of(angle: float, yaw: float) -> str:
    b = (angle + yaw) % (2 * math.pi)
    idx = int(math.floor(b / (math.pi / 2) + 0.5)) % 4
    return DIR_NAMES[idx]


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


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    q = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return q / np.linalg.norm(q)


# --------------------------------------------------------------------------- #
# Main merge
# --------------------------------------------------------------------------- #
def merge_nested(args) -> None:
    """Merge the demo leg: rebuild the 14-link chain from a flat o2r export.

    ``onshape-to-robot`` correctly places the two halves of every station but
    writes all 14 stations as free bodies, because the demo assembly's joints
    are not named ``dof_*``.  This path groups the bodies by their Onshape
    occurrence, nests them into a chain, adds ball joints at the gap midpoints,
    and then injects the same sites / tendons / servos / rod spring as the
    original leg.
    """
    stations = load_nested_stations(
        args.document, args.workspace, args.assembly,
        args.bore_r, args.inner_r, args.outer_r,
    )

    import mujoco  # noqa: F401  (validates the final XML)

    tree = ET.parse(args.xml)
    root = tree.getroot()
    wb = root.find("worldbody")
    bodies = collect_bodies(wb)
    if len(bodies) != len(stations):
        raise RuntimeError(f"{len(bodies)} bodies vs {len(stations)} stations")

    # Match o2r bodies to Onshape stations by name, with a position fallback.
    by_name = {b.get("name"): b for b in bodies}

    def signature(translations) -> tuple:
        return tuple(sorted(
            (round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3))
            for t in translations
        ))

    def body_signature(body) -> tuple:
        pts = set()
        for geom in body.findall("geom"):
            p = [float(x) for x in geom.get("pos", "0 0 0").split()]
            pts.add((round(p[0], 3), round(p[1], 3), round(p[2], 3)))
        return tuple(sorted(pts))

    ordered = []
    for s in stations:
        body = by_name.get(s["body"])
        if body is None:
            want = signature([T[:3, 3] for T, _ in s["parts"]])
            candidates = [b for b in bodies if body_signature(b) == want]
            if candidates:
                body = candidates[0]
        if body is None:
            raise RuntimeError(f"no o2r body for station {s['name']}")
        ordered.append(body)
    if len({id(b) for b in ordered}) != len(stations):
        raise RuntimeError("body/station mapping is not one-to-one")

    # --- untangle the flat body list -------------------------------------- #
    # Remove joints/inertials before detaching the bodies; root.iter() can no
    # longer see them once the bodies are out of the worldbody.
    for body in bodies:
        for child in body.findall("freejoint") + body.findall("inertial"):
            body.remove(child)
        wb.remove(body)

    # --- nest top -> bottom, re-express geoms around each disk centre ----- #
    for i, (s, body) in enumerate(zip(stations, ordered)):
        ctr = s["center"]
        for child in body.findall("geom") + body.findall("site"):
            p = np.array([float(x) for x in child.get("pos", "0 0 0").split()])
            child.set("pos", _vec(p - ctr))

        if i == 0:
            wb.append(body)
            body.set("pos", "0 0 0")
            body.set("quat", "1 0 0 0")
            if not args.fix_base:
                ET.SubElement(body, "freejoint", {"name": "root_free"})
        else:
            pc = stations[i - 1]["center"]
            ordered[i - 1].append(body)
            body.set("pos", _vec(ctr - pc))
            body.set("quat", "1 0 0 0")
            mid = 0.5 * (ctr + pc)
            joint = ET.Element("joint", {
                "name": f"{i - 1:02d}",
                "type": "ball",
                "pos": _vec(mid - ctr),
            })
            body.insert(0, joint)

    # --- tendon sites ------------------------------------------------------ #
    # The demo hip plate is clocked 45 deg relative to the disk stacks.  Use
    # the old rule (first station's hole nearest +45 deg -> +X) so all four
    # holes at every station still land on four distinct names.
    ref = [math.atan2(p[1] - stations[0]["center"][1],
                      p[0] - stations[0]["center"][0])
           for kind, p in stations[0]["holes"] if kind == "inner"]
    front = min(ref, key=lambda a: circular_dist(a, math.radians(45)))
    yaw = -front
    print(f"* tendon yaw {math.degrees(yaw):.2f} deg")

    site_world: dict[tuple, np.ndarray] = {}
    seen: set[tuple] = set()
    for i, (s, body) in enumerate(zip(stations, ordered)):
        ctr = s["center"]
        local_seen: set[tuple] = set()
        for kind, p in s["holes"]:
            ang = math.atan2(p[1] - ctr[1], p[0] - ctr[0])
            d = _dir_of(ang, yaw)
            key = (kind, d)
            if key in local_seen:
                continue
            local_seen.add(key)
            name = f"tendon_{kind}_{d}_{i:02d}"
            ET.SubElement(body, "site", {
                "name": name,
                "pos": _vec(p - ctr),
                "size": "0.0015",
                "rgba": "0.9 0.1 0.1 1" if kind == "inner" else "0.1 0.4 0.9 1",
            })
            site_world[(i, kind, d)] = p

    # --- tip at the bottom centre of the foot adaptor ---------------------- #
    last = len(stations) - 1
    foot = stations[last]
    tip_local = np.array([0.0, 0.0, foot["zmin"] - foot["center"][2]])
    ET.SubElement(ordered[last], "site", {
        "name": "tip",
        "pos": _vec(tip_local),
        "size": "0.004",
        "rgba": "0.1 0.9 0.1 1",
    })
    print(f"* tip {foot['zmin']:.4f} m below the hip")

    # --- tendons ----------------------------------------------------------- #
    tendon = ET.SubElement(root, "tendon")
    tendon_specs = []
    for d in DIR_NAMES:
        keys = [(i, "inner", d) for i in range(len(stations))
                if (i, "inner", d) in site_world]
        tendon_specs.append((f"tendon_inner_{d}",
                             [f"tendon_{k[1]}_{k[2]}_{k[0]:02d}" for k in keys],
                             polyline_len(keys, site_world)))
    for d in DIR_NAMES:
        keys = [(i, "outer", d) for i in range(len(stations))
                if (i, "outer", d) in site_world]
        if len(keys) >= 2:
            tendon_specs.append((f"tendon_outer_{d}",
                                 [f"tendon_{k[1]}_{k[2]}_{k[0]:02d}" for k in keys],
                                 polyline_len(keys, site_world)))
    for name, sites, _ in tendon_specs:
        sp = ET.SubElement(tendon, "spatial", {"name": name, "width": "0.001"})
        for s in sites:
            ET.SubElement(sp, "site", {"site": s})
        print(f"* {name}: {len(sites)} sites")

    # --- actuators --------------------------------------------------------- #
    actuator = ET.SubElement(root, "actuator")
    if args.servo:
        fmax = args.servo_torque / args.spool_radius
        stroke = args.spool_radius * args.servo_travel
        print(f"* servo: torque={args.servo_torque:.2f}Nm "
              f"spool={args.spool_radius*1000:.1f}mm -> Fmax={fmax:.1f}N "
              f"stroke={stroke*1000:.1f}mm travel={math.degrees(args.servo_travel):.0f}deg")
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
    cz = [s["center"][2] for s in stations]
    pitch = float(np.mean(np.abs(np.diff(cz))))
    I = math.pi * args.rod_d ** 4 / 64.0
    k = args.rod_E * I / pitch
    print(f"* rod: d={args.rod_d*1000:.2f}mm E={args.rod_E/1e9:.1f}GPa "
          f"L={pitch*1000:.2f}mm -> k={k:.1f} N.m/rad per joint")
    for j in root.iter("joint"):
        if j.get("type") == "ball":
            j.set("stiffness", f"{k:.6g}")
            j.set("damping", "0.5")
            j.set("armature", "0.01")

    # --- mass from PLA density --------------------------------------------- #
    for default in root.iter("default"):
        cls = default.get("class")
        for geom in default.findall("geom"):
            if cls == "visual":
                geom.set("density", "0")
            elif cls == "collision":
                geom.set("density", str(args.density))

    # --- optional yaw ------------------------------------------------------ #
    if args.rotate:
        root_body = ordered[0]
        q = np.array([float(x) for x in root_body.get("quat", "1 0 0 0").split()])
        yq = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        root_body.set("quat", _vec(quat_mul(yq, q)))
        print(f"* yaw {math.degrees(yaw):.2f} deg")

    ET.indent(tree, space="  ")
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"* wrote {args.out}")

    m = mujoco.MjModel.from_xml_path(args.out)
    total_mass = sum(float(m.body_mass[i]) for i in range(m.nbody))
    print(f"* check: bodies={m.nbody} joints={m.njnt} sites={m.nsite} "
          f"tendons={m.ntendon} actuators={m.nu} mass={total_mass*1000:.1f} g")


def polyline_len(keys: list[tuple], site_world: dict) -> float:
    return float(sum(
        np.linalg.norm(site_world[keys[j + 1]] - site_world[keys[j]])
        for j in range(len(keys) - 1)
    ))


def merge(args) -> None:
    if args.nested:
        merge_nested(args)
        return
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

    # --- drawing tip at the foot centre ----------------------------------- #
    last = len(stations) - 1
    xpos = data.xpos[last + 1].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.xquat[last + 1])
    R = R.reshape(3, 3)
    inner_pts = [p for kind, _, p in holes_world(stations[last]) if kind == "inner"]
    if inner_pts:
        centre = np.mean(inner_pts, axis=0)
        local = R.T @ (centre - xpos)
        ET.SubElement(bodies[last], "site", {
            "name": "tip",
            "pos": " ".join(f"{v:.9g}" for v in local),
            "size": "0.004",
            "rgba": "0.1 0.9 0.1 1",
        })
        print(f"* tip site at {centre}")

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
        q = np.array([float(x) for x in root_body.get("quat", "1 0 0 0").split()])
        yq = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        root_body.set("quat", " ".join(f"{v:.9g}" for v in quat_mul(yq, q)))
        print(f"* yaw {math.degrees(yaw):.2f} deg")

    # --- optional 180 deg flip about world X so the tip points up ---------- #
    if args.tip_up:
        root_body = bodies[0]
        q = np.array([float(x) for x in root_body.get("quat", "1 0 0 0").split()])
        flip = np.array([0.0, 1.0, 0.0, 0.0])   # 180 deg about X
        root_body.set("quat", " ".join(f"{v:.9g}" for v in quat_mul(flip, q)))
        print("* tip-up: flipped 180 deg about world X")

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
    p.add_argument("--nested", action="store_true",
                   help="demo leg: rebuild a chain from a flat nested assembly export")
    p.add_argument("--bore-r", type=float, default=0.004862,
                   help="backbone bore radius (m) for --nested")
    p.add_argument("--inner-r", type=float, default=0.0191,
                   help="inner tendon ring radius (m) for --nested")
    p.add_argument("--outer-r", type=float, default=0.0291,
                   help="outer tendon ring radius (m) for --nested")
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
    p.add_argument("--tip-up", action="store_true",
                   help="flip the model 180 deg about world X so the foot points up")
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    p.add_argument("--free-base", dest="fix_base", action="store_false")
    p.set_defaults(rotate=True, fix_base=True, tip_up=False)
    args = p.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
