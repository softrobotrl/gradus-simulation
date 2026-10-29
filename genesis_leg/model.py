"""Build the leg and board scene from the existing robot configuration."""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from pipeline.config import RobotConfig
from pipeline.stl_to_xml import stl_to_xml, write


DEFAULT_CONFIG = "pipeline/configs/opencr_tdcr_spatial.yaml"
BUILD_ROOT = Path(__file__).resolve().parents[1] / "pipeline/build/genesis"


def build_scene(config=DEFAULT_CONFIG, out_root=None):
    """Write the scene XML and cable routes, then return them with the robot config."""
    cfg = RobotConfig.load(config)
    if out_root is None:
        # Training and evaluation may start together. Give each process its own
        # files so neither can import a partially written scene from the other.
        BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        out_root = tempfile.mkdtemp(prefix="scene-", dir=BUILD_ROOT)
    xml = stl_to_xml(config, Path(out_root))
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = ET.parse(xml).getroot()
    tip_id = model.site("tip").id
    tip_body = model.body(int(model.site_bodyid[tip_id])).name
    tip_local = model.site_pos[tip_id].tolist()

    # Match the MuJoCo task: the straight leg lifts the board 3 mm off its rails.
    rail_z = float(data.site_xpos[tip_id, 2]) + 0.004 - 0.003
    top = root.find(f".//body[@name='{tip_body}']")
    ET.SubElement(
        top, "geom",
        name="pad", type="sphere", size="0.004", mass="0",
        pos=" ".join(map(str, tip_local)),
        contype="1", conaffinity="0",
        friction="0.8 0.005 0.0001", solref="0.005 1", rgba="0.1 0.1 0.1 1",
    )
    board = ET.SubElement(
        root.find("worldbody"), "body",
        name="board", pos=f"0 0 {rail_z + 0.006}",
    )
    for name, axis in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
        attrs = dict(name=f"board_{name}", type="slide", axis=axis, damping="0.5")
        if name == "z":
            attrs.update(limited="true", range="0 0.03", solreflimit="0.005 1")
        else:
            attrs.update(frictionloss="0.2")
        ET.SubElement(board, "joint", **attrs)
    ET.SubElement(
        board, "geom",
        name="board", type="box", size="0.2 0.075 0.006", mass="0.1",
        contype="0", conaffinity="1",
        friction="0.8 0.005 0.0001", solref="0.005 1", rgba="0.75 0.55 0.3 1",
    )

    cables = []
    for tendon in root.find("tendon"):
        sites = [model.site(node.attrib["site"]).id for node in tendon]
        cables.append({
            "rest_length": float(tendon.attrib["range"].split()[1]),
            "bodies": [model.body(int(model.site_bodyid[s])).name for s in sites],
            "offsets": model.site_pos[sites].tolist(),
        })
    # env.py supplies the cable forces and motor torques that Genesis won't import.
    root.remove(root.find("tendon"))
    root.remove(root.find("actuator"))
    # Ball-joint limits are checked by the environment instead.
    for joint in root.findall(".//joint[@type='ball']"):
        joint.set("limited", "false")
    scene_xml = xml.with_name("board_scene.xml")
    write(root, scene_xml)
    metadata = {
        "cables": cables,
        "tip_body": tip_body,
        "tip_local": tip_local,
        "rail_z": rail_z,
        "n_disks": len(cfg.disk_sequence()),
        "config": str(Path(config).resolve()),
    }
    scene_xml.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return scene_xml.resolve(), metadata, cfg
