"""Stage 1: STL parts + robot config -> MuJoCo MJCF (robot.xml) with joints, tendons and servos.

Usage: python -m pipeline.stl_to_xml pipeline/configs/opencr_tdcr_spatial.yaml [--out DIR]
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from pipeline.config import RobotConfig
from pipeline.mesh_features import DiskFeatures, analyse_disk, load_stl

DEFAULT_OUT = Path(__file__).parent / "build"
NO_COLLIDE = {"contype": "0", "conaffinity": "0"}


def fmt(*values: float) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def joint_stiffness(cfg: RobotConfig) -> float:
    joint, backbone = cfg.chain.joint, cfg.chain.backbone
    if joint.stiffness is not None:
        return joint.stiffness
    if backbone is None:
        raise ValueError("chain.joint.stiffness is null but no backbone is given to derive it from")
    second_moment = math.pi / 64 * (backbone.outer_diameter**4 - backbone.inner_diameter**4)
    return backbone.youngs_modulus * second_moment / cfg.chain.spacing


def analyse_parts(cfg: RobotConfig, mesh_dir: Path) -> dict[str, DiskFeatures]:
    features = {}
    for stl in dict.fromkeys(cfg.disk_sequence()):
        feat = analyse_disk(load_stl(cfg.stl_dir / stl, cfg.units))
        if cfg.chain.tendon_holes != "auto":
            feat.holes = np.array(cfg.chain.tendon_holes)
        feat.mesh.export(mesh_dir / stl)
        features[stl] = feat
    counts = {stl: len(f.holes) for stl, f in features.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"disks disagree on tendon-hole count: {counts}")
    for part in cfg.static_parts:
        load_stl(cfg.stl_dir / part.stl, cfg.units).export(mesh_dir / part.stl)
    return features


def build_mjcf(cfg: RobotConfig, features: dict[str, DiskFeatures]) -> ET.Element:
    chain, tendons, servo = cfg.chain, cfg.tendons, cfg.servo
    disks = cfg.disk_sequence()
    n_tendons = len(features[disks[0]].holes)

    root = ET.Element("mujoco", model=cfg.name)
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    ET.SubElement(root, "option", timestep=f"{cfg.sim.timestep}", integrator=cfg.sim.integrator)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "scale", jointlength="0.01", jointwidth="0.002", framelength="0.02")

    asset = ET.SubElement(root, "asset")
    for stl in dict.fromkeys(disks + [p.stl for p in cfg.static_parts]):
        ET.SubElement(asset, "mesh", name=Path(stl).stem, file=stl)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 0.6", dir="0 0.5 -1", directional="true")
    floor_z = chain.base_height - tendons.pulley_depth - 0.01
    ET.SubElement(world, "geom", name="floor", type="plane", size="0.3 0.3 0.01",
                  pos=fmt(0, 0, floor_z), rgba="0.9 0.9 0.9 1", **NO_COLLIDE)
    for i, part in enumerate(cfg.static_parts):
        ET.SubElement(world, "geom", name=f"static{i}", type="mesh", mesh=Path(part.stl).stem,
                      pos=fmt(*part.pos), euler=fmt(*np.deg2rad(part.euler_deg)),
                      rgba=fmt(*part.rgba), mass="0", **NO_COLLIDE)

    # Disk chain: disk 0 is welded to the base, every further disk hangs on a ball joint.
    stiffness = joint_stiffness(cfg)
    bb = chain.backbone
    parent = world
    for i, stl in enumerate(disks):
        pos = (0, 0, chain.base_height) if i == 0 else (0, 0, chain.spacing)
        body = ET.SubElement(parent, "body", name=f"disk{i}", pos=fmt(*pos))
        if i > 0:
            j = chain.joint
            ET.SubElement(body, "joint", name=f"joint{i}", type="ball",
                          pos=fmt(0, 0, j.pivot_offset * chain.spacing),
                          stiffness=f"{stiffness:.6g}", damping=f"{j.damping:.6g}",
                          armature=f"{j.armature:.6g}", frictionloss=f"{j.frictionloss:.6g}",
                          range=fmt(0, math.radians(j.range_deg)))
            if bb is not None:
                area = math.pi / 4 * (bb.outer_diameter**2 - bb.inner_diameter**2)
                ET.SubElement(body, "geom", name=f"backbone{i}", type="capsule",
                              fromto=fmt(0, 0, 0, 0, 0, -chain.spacing), size=fmt(bb.outer_diameter / 2),
                              mass=f"{area * chain.spacing * bb.density:.6g}", rgba="0.3 0.3 0.35 1",
                              **NO_COLLIDE)
        ET.SubElement(body, "geom", name=f"disk{i}_mesh", type="mesh", mesh=Path(stl).stem,
                      density=f"{chain.density}", rgba="0.85 0.55 0.2 1", **NO_COLLIDE)
        if chain.extra_mass_per_disk > 0:
            ET.SubElement(body, "geom", name=f"disk{i}_extra", type="sphere", size="0.001",
                          mass=f"{chain.extra_mass_per_disk}", group="3", **NO_COLLIDE)
        for k, (x, y) in enumerate(features[stl].holes):
            ET.SubElement(body, "site", name=f"disk{i}_hole{k}", pos=fmt(x, y, 0), size="0.0006")
        if i == len(disks) - 1:
            ET.SubElement(body, "site", name="tip", pos=fmt(0, 0, features[stl].thickness / 2),
                          size="0.0015", rgba="0 0.6 0 1")
        parent = body

    # Servo side: cable runs from each bottom hole straight down to a pulley, then radially out to a
    # carriage. The carriage slide stands in for the drum: 1 rad of drum rotation = drum_radius of cable.
    pulley_z = chain.base_height - tendons.pulley_depth
    reflected = servo.rotor_inertia * servo.gear_ratio**2 / servo.drum_radius**2
    for k, (x, y) in enumerate(features[disks[0]].holes):
        u = np.array([x, y]) / np.hypot(x, y)
        ET.SubElement(world, "site", name=f"pulley{k}", pos=fmt(x, y, pulley_z), size="0.002")
        carriage = ET.SubElement(world, "body", name=f"carriage{k}",
                                 pos=fmt(*(tendons.carriage_radius * u), pulley_z))
        ET.SubElement(carriage, "joint", name=f"cable{k}", type="slide", axis=fmt(*u, 0),
                      armature=f"{reflected:.6g}")
        ET.SubElement(carriage, "geom", type="box", size="0.004 0.004 0.004", mass="0.001",
                      rgba="0.2 0.4 0.8 1", **NO_COLLIDE)
        ET.SubElement(carriage, "site", name=f"carriage{k}", size="0.001")

    tendon_el = ET.SubElement(root, "tendon")
    for k in range(n_tendons):
        spatial = ET.SubElement(tendon_el, "spatial", name=f"tendon{k}", width="0.0004",
                                rgba="0.8 0.1 0.1 1", limited="true", range="0 1",
                                solreflimit=fmt(tendons.limit_timeconst, 1))
        for site in [f"carriage{k}", f"pulley{k}"] + [f"disk{i}_hole{k}" for i in range(len(disks))]:
            ET.SubElement(spatial, "site", site=site)

    actuator = ET.SubElement(root, "actuator")
    for k in range(n_tendons):
        ET.SubElement(actuator, "position", name=f"servo{k}", joint=f"cable{k}",
                      gear=f"{1 / servo.drum_radius:.6g}", kp=f"{servo.kp}", kv=f"{servo.kv}",
                      ctrlrange=fmt(*servo.angle_range), forcerange=fmt(-servo.max_torque, servo.max_torque))
    return root


def set_rest_lengths(root: ET.Element, xml_path: Path, slack: float) -> list[float]:
    """Cable length = path length in the straight pose at servo angle 0 (taut, no pretension)."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lengths = [float(l) + slack for l in data.ten_length]
    for el, length in zip(root.iter("spatial"), lengths):
        el.set("range", fmt(0, length))
    return lengths


def write(root: ET.Element, path: Path) -> None:
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def stl_to_xml(config_path: str | Path, out_root: Path = DEFAULT_OUT) -> Path:
    cfg = RobotConfig.load(config_path)
    out = out_root / cfg.name
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    features = analyse_parts(cfg, out / "meshes")
    root = build_mjcf(cfg, features)
    xml_path = out / "robot.xml"
    write(root, xml_path)
    rest = set_rest_lengths(root, xml_path, cfg.tendons.slack)
    write(root, xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    report = {
        "disks": {stl: {"thickness_mm": round(f.thickness * 1e3, 3), "outer_radius_mm": round(f.outer_radius * 1e3, 3),
                        "hole_radius_mm": round(f.hole_radius * 1e3, 3),
                        "tendon_holes_mm": (f.holes * 1e3).round(3).tolist()} for stl, f in features.items()},
        "n_disks": len(cfg.disk_sequence()),
        "n_tendons": len(rest),
        "joint_stiffness_Nm_per_rad": joint_stiffness(cfg),
        "tendon_rest_length_m": [round(l, 6) for l in rest],
        "chain_mass_kg": float(model.body_subtreemass[model.body("disk0").id]),
    }
    (out / "features.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(yaml.safe_dump(report, sort_keys=False))
    print(f"wrote {xml_path}")
    return xml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    stl_to_xml(args.config, args.out)
