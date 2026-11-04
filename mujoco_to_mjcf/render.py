#!/usr/bin/env python3
"""Render the merged leg model to PNGs (works headless under xvfb + glfw).

    xvfb-run -a env MUJOCO_GL=glfw python -m mujoco_to_mjcf.render \
        --xml urdf/gradus/sim/scene.xml --out urdf/gradus/sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco

POSES = {
    "rest": None,
    "inner_front": ("act_tendon_inner_front", 200.0, 2500),
    "inner_back": ("act_tendon_inner_back", 200.0, 2500),
    "inner_left": ("act_tendon_inner_left", 200.0, 2500),
    "outer_front": ("act_tendon_outer_front", 400.0, 2500),
}


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
        mujoco.mj_forward(model, data)
        if spec is not None:
            actuator, force, steps = spec
            data.ctrl[model.actuator(actuator).id] = force
            for _ in range(steps):
                mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        path = args.out / f"leg_pose_{name}.png"
        imageio.imwrite(path, renderer.render())
        print(path)


if __name__ == "__main__":
    main()
