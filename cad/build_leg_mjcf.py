#!/usr/bin/env python3
"""One-command pipeline: Onshape assembly -> onshape-to-robot -> merged MuJoCo MJCF.

Output (``--out``, default ``urdf/gradus/sim``)::

    leg.xml        bodies, ball joints, 84 tendon sites, 8 spatial tendons,
                   8 tension-only motors, backbone stiffness, real mass
    scene.xml      include + floor/light (mujoco.viewer ready)
    assets/*.stl   meshes referenced by leg.xml

The clean ``Leg-Sim`` assembly is reused when it already exists; pass
``--rebuild`` to create a fresh one from the original assembly.

Examples::

    python -m cad.build_leg_mjcf
    python -m cad.build_leg_mjcf --rebuild
    python -m cad.build_leg_mjcf --assembly <element-id> --out /tmp/leg

Everything geometric (station order, hole radii/angles, link lengths) is read
from Onshape; ``cad/leg_config.json`` only holds document IDs and material
defaults.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cad.build_sim_assembly import build, ensure, is_valid  # noqa: E402

CONFIG_PATH = Path(__file__).with_name("leg_config.json")

SCENE = """<mujoco model="leg_scene">
  <include file="leg.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="140" elevation="-15"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="ground" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="ground" texture="ground" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 -0.65" type="plane" material="ground"/>
  </worldbody>
</mujoco>
"""


def load_config(path: Path) -> dict:
    if path and path.exists():
        return json.loads(path.read_text())
    return {}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def o2r_bin() -> str:
    local = Path(sys.executable).with_name("onshape-to-robot")
    if local.exists():
        return str(local)
    found = shutil.which("onshape-to-robot")
    if not found:
        raise SystemExit("onshape-to-robot is not installed in this environment")
    return found


def validate(xml: Path) -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(300):
        mujoco.mj_step(m, d)
    if np.isnan(d.qpos).any():
        raise RuntimeError("merged model produced NaN state")
    mass = sum(float(x) for x in m.body_mass) * 1000.0
    print(
        f"* validate: bodies={m.nbody} joints={m.njnt} sites={m.nsite} "
        f"tendons={m.ntendon} actuators={m.nu} mass={mass:.0f} g"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "urdf/gradus/sim")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--assembly", default=None, help="use this Leg-Sim element id")
    p.add_argument("--rebuild", action="store_true",
                   help="create a fresh Leg-Sim assembly from the original")
    p.add_argument("--keep-work", action="store_true",
                   help="keep the onshape-to-robot intermediate in <out>/_o2r")
    p.add_argument("--rod-d", type=float, default=None, help="backbone diameter (m)")
    p.add_argument("--rod-E", type=float, default=None, help="backbone E (Pa)")
    p.add_argument("--density", type=float, default=None, help="PLA density (kg/m^3)")
    p.add_argument("--tendon-fmax", type=float, default=None, help="motor force limit (N)")
    p.add_argument("--servo", action="store_true", help="model servos instead of force motors")
    p.add_argument("--motors", action="store_true", help="force-motor model even if config uses servos")
    p.add_argument("--servo-torque", type=float, default=None, help="servo stall torque (N.m)")
    p.add_argument("--spool-radius", type=float, default=None, help="tendon spool radius (m)")
    p.add_argument("--servo-travel", type=float, default=None, help="max spool travel (rad)")
    p.add_argument("--servo-kp", type=float, default=None, help="servo position gain")
    p.add_argument("--servo-kv", type=float, default=None, help="servo velocity gain")
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--free-base", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    did = cfg["document_id"]
    wid = cfg["workspace_id"]
    ps = cfg["partstudio_id"]
    src = cfg["source_assembly_id"]
    name = cfg.get("sim_assembly_name", "Leg-Sim")

    # --- which Leg-Sim assembly -------------------------------------------- #
    asm = args.assembly or cfg.get("sim_assembly_id")
    if args.rebuild:
        asm = build(did, wid, ps, src, name)
    elif not (asm and is_valid(did, wid, asm)):
        asm = ensure(did, wid, ps, src, name)
    else:
        print(f"* reusing {name}: {asm}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not args.keep_work:
        shutil.rmtree(out / "_o2r", ignore_errors=True)

    # --- onshape-to-robot --------------------------------------------------- #
    work = Path(tempfile.mkdtemp(prefix="leg_o2r_", dir="/tmp/opencode"))
    try:
        config = {
            "url": f"https://cad.onshape.com/documents/{did}/w/{wid}/e/{asm}",
            "output_format": "mujoco",
            "output_filename": "leg_base",
            "assets_directory": "assets",
            "ignore_limits": True,
        }
        (work / "config.json").write_text(json.dumps(config, indent=2))
        run([o2r_bin(), str(work)])

        base_xml = work / "leg_base.xml"
        if not base_xml.exists():
            raise RuntimeError("onshape-to-robot did not write leg_base.xml")

        # assets must be next to the merged XML before validation
        dst_assets = out / "assets"
        if dst_assets.exists():
            shutil.rmtree(dst_assets)
        shutil.copytree(work / "assets", dst_assets)

        # --- merge ---------------------------------------------------------- #
        merge = [
            sys.executable, "-m", "cad.merge_mjcf",
            "--xml", str(base_xml),
            "--out", str(out / "leg.xml"),
            "--document", did,
            "--workspace", wid,
            "--assembly", asm,
            "--partstudio", ps,
            "--rod-d", str(args.rod_d if args.rod_d is not None else cfg["rod_diameter_m"]),
            "--rod-E", str(args.rod_E if args.rod_E is not None else cfg["rod_youngs_modulus_pa"]),
            "--density", str(args.density if args.density is not None else cfg["pla_density_kg_m3"]),
            "--tendon-fmax", str(args.tendon_fmax if args.tendon_fmax is not None else cfg["tendon_force_max_n"]),
            "--servo-torque", str(args.servo_torque if args.servo_torque is not None else cfg["servo_torque_nm"]),
            "--spool-radius", str(args.spool_radius if args.spool_radius is not None else cfg["spool_radius_m"]),
            "--servo-travel", str(args.servo_travel if args.servo_travel is not None else cfg["servo_travel_rad"]),
            "--servo-kp", str(args.servo_kp if args.servo_kp is not None else cfg["servo_kp"]),
            "--servo-kv", str(args.servo_kv if args.servo_kv is not None else cfg["servo_kv"]),
        ]
        if (args.servo or cfg.get("use_servos", False)) and not args.motors:
            merge.append("--servo")
        if args.no_rotate or not cfg.get("rotate_to_axes", True):
            merge.append("--no-rotate")
        if args.free_base or not cfg.get("fix_base", True):
            merge.append("--free-base")
        run(merge)

        (out / "scene.xml").write_text(SCENE)
        validate(out / "leg.xml")

        if args.keep_work:
            dst = out / "_o2r"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(work, dst)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"* done: {out/'leg.xml'}  (+ scene.xml, assets/)")


if __name__ == "__main__":
    main()
