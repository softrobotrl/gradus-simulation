"""Render a trained tracer drawing a shape to GIF (no JAX/CUDA needed).

Example::

    python -m mujoco_leg.render_shape --shape star --out logs/mujoco_leg/tracer_final/draw_star.gif
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from mujoco_leg.view import (
    add_sphere,
    load_checkpoint,
    make_shape,
    observation,
    policy_action,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path,
                   default=Path("logs/mujoco_leg/tracer_final/final.pkl"))
    p.add_argument("--shape", default="star",
                   choices=["star", "circle", "square", "figure_eight"])
    p.add_argument("--view", default="top", choices=["top", "perspective"])
    p.add_argument("--radius", type=float, default=None,
                   help="shape radius in metres (default: training radius)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--size", type=int, default=480)
    args = p.parse_args()

    cfg, params = load_checkpoint(args.ckpt)
    pi = params["pi"]
    radius = args.radius if args.radius is not None else cfg["shape_radius"]
    shape = make_shape(args.shape, cfg["shape_points"], radius)
    out = args.out or Path(f"logs/mujoco_leg/tracer_final/draw_{args.shape}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path("urdf/gradus/tracer/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    l0 = model.actuator_ctrlrange[:, 1].copy()
    stroke = model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]
    fmax = -model.actuator_forcerange[:, 0].copy()
    tip_id = model.site("tip").id
    data.ctrl[:] = l0
    mujoco.mj_forward(model, data)

    phase = 0.0
    prev_tip = data.site_xpos[tip_id].copy()
    tip_z = float(prev_tip[2])
    z_draw = tip_z + 0.02
    dt = float(model.opt.timestep) * cfg["n_substeps"]
    trail = deque(maxlen=args.steps)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if args.view == "top":
        cam.lookat[:] = [0.0, 0.0, tip_z]
        cam.distance = 0.40
        cam.azimuth = 90.0
        cam.elevation = -89.0
    else:
        cam.lookat[:] = [0.0, 0.0, 0.50]
        cam.distance = 0.85
        cam.azimuth = 140.0
        cam.elevation = 20.0

    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    frames = []
    for step in range(args.steps):
        obs = observation(model, data, shape, phase, prev_tip, l0, stroke, fmax, tip_id, cfg)
        action = policy_action(pi, obs)
        pull = 0.5 * (action + 1.0)
        target_ctrl = l0 - stroke * pull
        data.ctrl[:] = np.clip(0.5 * data.ctrl + 0.5 * target_ctrl, l0 - stroke, l0)
        for _ in range(cfg["n_substeps"]):
            mujoco.mj_step(model, data)

        tip = data.site_xpos[tip_id].copy()
        phase = (phase + dt / cfg["period"]) % 1.0
        prev_tip = tip
        trail.append(tip.copy())

        if step % args.every == 0:
            renderer.update_scene(data, camera=cam)
            scn = renderer.scene
            for q in shape[::2]:
                add_sphere(scn, (q[0], q[1], z_draw), 0.0012, (0.2, 0.45, 1.0, 0.55))
            for q in trail:
                add_sphere(scn, (q[0], q[1], q[2] + 0.02), 0.0008, (1.0, 0.55, 0.1, 0.85))
            frames.append(renderer.render().copy())

    renderer.close()
    imageio.mimsave(out, frames, fps=25)
    print(f"* wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
