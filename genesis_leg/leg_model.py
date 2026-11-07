"""Turn a merged leg.xml (mujoco_to_mjcf) into a model Genesis can load, plus its tendon routes.

Genesis loads the bodies, ball joints, stiffness and mass from MJCF but not the spatial tendons or tendon actuators.
Those are read here from the compiled MuJoCo model, so leg.py can apply them as forces.
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# Sections that refer to the removed tendons and actuators.
REMOVED_SECTIONS = ("tendon", "actuator", "sensor", "keyframe")


def read_routes(model):
    """Each tendon's route and servo settings, in actuator order."""
    tendons = []
    for a in range(model.nu):
        name = model.actuator(a).name
        is_tendon = model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON
        if not is_tendon or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
            raise ValueError(f"{name}: expected a <position> servo on a tendon; rebuild leg.xml without --motors")
        low, high = model.actuator_forcerange[a]
        if not model.actuator_forcelimited[a] or low >= 0 or high != 0:
            raise ValueError(f"{name}: expected a tension-only forcerange [-Fmax, 0], got [{low}, {high}]")
        if model.actuator_gear[a, 0] != 1:
            raise ValueError(f"{name}: expected gear 1, got {model.actuator_gear[a, 0]}")
        t = model.actuator_trnid[a, 0]
        wraps = range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t])
        if any(model.wrap_type[w] != mujoco.mjtWrap.mjWRAP_SITE for w in wraps):
            raise ValueError(f"{model.tendon(t).name}: only site-to-site routes are supported")
        sites = [int(model.wrap_objid[w]) for w in wraps]
        tendons.append({
            "name": model.tendon(t).name,
            "bodies": [model.body(int(model.site_bodyid[s])).name for s in sites],
            "offsets": model.site_pos[sites].tolist(),
            "kp": float(model.actuator_gainprm[a, 0]),
            "kv": float(-model.actuator_biasprm[a, 2]),
            "fmax": float(-low),
            "ctrl_low": float(model.actuator_ctrlrange[a, 0]),
            "ctrl_high": float(model.actuator_ctrlrange[a, 1]),
        })
    if not tendons:
        raise ValueError("leg.xml has no tendon actuators")
    return tendons


def find_tip(model, tendons):
    """The `tip` site if the model has one, otherwise the centre of the holes on the last station."""
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    if tip >= 0:
        return model.body(int(model.site_bodyid[tip])).name, model.site_pos[tip].tolist()
    body = max(tendons, key=lambda tendon: len(tendon["bodies"]))["bodies"][-1]
    ends = [tendon["offsets"][-1] for tendon in tendons if tendon["bodies"][-1] == body]
    return body, np.mean(ends, axis=0).tolist()


def _replace_text(path, text):
    # Runs starting together may share this directory; never let one read a partly written file.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def prepare_leg(xml):
    """Write <name>_genesis.xml and <name>_routes.json next to `xml`, so mesh paths still resolve. Return (path, routes)."""
    xml = Path(xml).resolve()
    model = mujoco.MjModel.from_xml_path(str(xml))
    tendons = read_routes(model)
    tip_body, tip_local = find_tip(model, tendons)

    root = ET.parse(xml).getroot()
    for section in REMOVED_SECTIONS:
        for element in root.findall(section):
            root.remove(element)
    # No collision in this model: nothing to touch yet, and mesh collision is expensive to prepare and run.
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    # Genesis ignores ball-joint limits and warns about each one.
    for joint in root.iter("joint"):
        if joint.get("type") == "ball":
            joint.set("limited", "false")

    routes = {
        "source": str(xml),
        "tendons": tendons,
        "tip_body": tip_body,
        "tip_local": tip_local,
        "timestep": float(model.opt.timestep),
        "gravity": model.opt.gravity.tolist(),
        "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
        "mass": float(model.body_mass.sum()),
    }
    out = xml.with_name(f"{xml.stem}_genesis.xml")
    ET.indent(root)
    _replace_text(out, ET.tostring(root, encoding="unicode") + "\n")
    _replace_text(xml.with_name(f"{xml.stem}_routes.json"), json.dumps(routes, indent=2) + "\n")
    return out, routes
