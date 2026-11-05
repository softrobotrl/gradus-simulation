#!/usr/bin/env python3
"""Render the merged leg model to PNGs (works headless under xvfb + glfw).

    xvfb-run -a env MUJOCO_GL=glfw python -m onshape_to_mjcf.render \
        --xml urdf/gradus/sim/scene.xml --out urdf/gradus/sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco

# Pose name -> (tendon, requested pull, simulation steps). The actuator is
# looked up from the tendon, so this works with both the default servo build
# (``servo_tendon_*``) and the force-motor build (``act_tendon_*``).
POSES = {
    "rest": None,
    "inner_front": ("tendon_inner_front", 200.0, 2500),
    "inner_back": ("tendon_inner_back", 200.0, 2500),
    "inner_left": ("tendon_inner_left", 200.0, 2500),
    "outer_front": ("tendon_outer_front", 400.0, 2500),
}


def actuator_for_tendon(model: mujoco.MjModel, tendon: str) -> int:
    """Return the actuator id driving ``tendon`` (servo or motor build)."""
    for i in range(model.nu):
        trnid = int(model.actuator_trnid[i, 0])
        if trnid < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, trnid)
        if name == tendon:
            return i
    raise KeyError(f"no actuator drives tendon {tendon!r}")


def actuator_name(model: mujoco.MjModel, i: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""


def is_servo(model: mujoco.MjModel, i: int) -> bool:
    return actuator_name(model, i).startswith("servo_")


def reset_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """No pull at rest: servos sit at their rest length, motors at zero."""
    for i in range(model.nu):
        data.ctrl[i] = model.actuator_ctrlrange[i, 1] if is_servo(model, i) else 0.0


def apply_pull(model: mujoco.MjModel, data: mujoco.MjData, i: int, pull: float) -> None:
    """Apply ``pull`` N to actuator ``i`` in either actuation mode."""
    if is_servo(model, i):
        lo, hi = model.actuator_ctrlrange[i]
        fmin, fmax = model.actuator_forcerange[i]
        limit = max(abs(fmin), abs(fmax)) or 1.0
        fraction = min(max(pull / limit, 0.0), 1.0)
        data.ctrl[i] = hi - (hi - lo) * fraction
    else:
        data.ctrl[i] = pull


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default="urdf/gradus/sim/scene.xml")
    p.add_argument("--out", type=Path, default=Path("urdf/gradus/sim"))
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    args.out.mkdir(parents=True, exist_ok=True)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, -0.30]
    cam.distance = 0.95
    cam.azimuth = 90.0
    cam.elevation = -5.0

    renderer = mujoco.Renderer(model, args.height, args.width)
    for name, spec in POSES.items():
        data = mujoco.MjData(model)
        reset_controls(model, data)
        mujoco.mj_forward(model, data)
        if spec is not None:
            tendon, pull, steps = spec
            i = actuator_for_tendon(model, tendon)
            apply_pull(model, data, i, pull)
            for _ in range(steps):
                mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        path = args.out / f"leg_pose_{name}.png"
        imageio.imwrite(path, renderer.render())
        print(path)


if __name__ == "__main__":
    main()
