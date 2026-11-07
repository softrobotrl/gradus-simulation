#!/usr/bin/env python3
"""Drive the leg with the tendons and watch it move.

The four inner (and four outer) tendons are pulled with phase-shifted
tension-only sinusoids, so the foot traces a circle.

Interactive (real display)::

    python -m mujoco_to_mjcf.play --xml urdf/gradus/sim/scene.xml

Headless GIF::

    xvfb-run -a env MUJOCO_GL=glfw python -m mujoco_to_mjcf.play \
        --xml urdf/gradus/sim/scene.xml \
        --record urdf/gradus/sim/leg_wave.gif --seconds 6

Larger circle / more tip lift (uses more of the 200 N servo limit)::

    MUJOCO_GL=egl python -m mujoco_to_mjcf.play \
        --xml urdf/gradus/walk/scene.xml \
        --servo-kp 8000 --servo-kv 80 --outer-scale 1.0 \
        --record urdf/gradus/walk/leg_circle_lift.gif --seconds 6
"""

from __future__ import annotations

import argparse
import math
import time

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np


def set_controls(model: mujoco.MjModel, data: mujoco.MjData,
                 t: float, amp: float, freq: float,
                 outer_scale: float = 0.45) -> None:
    """Circle-shaped tendon pull. All commands are >= 0 (tension only).

    Motor model:   ``ctrl`` is a tension in N.
    Servo model:   ``ctrl`` is a target tendon length (m); the position
                   actuator's ``forcerange`` makes it tension-only and
                   ``ctrlrange`` caps the stroke.
    """
    w = 2.0 * math.pi * freq * t
    fx, fy = math.cos(w), math.sin(w)
    inner = {
        "front": max(0.0, fx),
        "back": max(0.0, -fx),
        "left": max(0.0, fy),
        "right": max(0.0, -fy),
    }
    outer = {
        "front": outer_scale * max(0.0, math.cos(w + 0.6)),
        "back": outer_scale * max(0.0, -math.cos(w + 0.6)),
        "left": outer_scale * max(0.0, math.sin(w + 0.6)),
        "right": outer_scale * max(0.0, -math.sin(w + 0.6)),
    }
    servo = model.actuator(0).name.startswith("servo_")
    for kind, table in (("inner", inner), ("outer", outer)):
        for side, pull in table.items():
            name = f"{'servo' if servo else 'act'}_tendon_{kind}_{side}"
            i = model.actuator(name).id
            if servo:
                lo, hi = model.actuator_ctrlrange[i]
                data.ctrl[i] = hi - (hi - lo) * pull
            else:
                data.ctrl[i] = amp * pull


def apply_servo_gains(model: mujoco.MjModel,
                      kp: float | None, kv: float | None) -> None:
    """Override position-servo gains on a servo-driven model."""
    if kp is None and kv is None:
        return
    if not model.actuator(0).name.startswith("servo_"):
        raise SystemExit("--servo-kp/--servo-kv only apply to servo models")
    if kp is not None:
        model.actuator_gainprm[:, 0] = kp
        model.actuator_biasprm[:, 1] = -kp
    if kv is not None:
        model.actuator_biasprm[:, 2] = -kv


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, -0.30]
    cam.distance = 0.95
    cam.azimuth = 90.0
    cam.elevation = -5.0
    return cam


def play(args) -> None:
    model = mujoco.MjModel.from_xml_path(args.xml)
    apply_servo_gains(model, args.servo_kp, args.servo_kv)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, -0.30]
        viewer.cam.distance = 0.95
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -5.0
        t0 = time.time()
        while viewer.is_running():
            start = time.time()
            set_controls(model, data, time.time() - t0, args.amp, args.freq,
                         args.outer_scale)
            mujoco.mj_step(model, data)
            viewer.sync()
            left = model.opt.timestep - (time.time() - start)
            if left > 0:
                time.sleep(left)


def record(args) -> None:
    model = mujoco.MjModel.from_xml_path(args.xml)
    apply_servo_gains(model, args.servo_kp, args.servo_kv)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = camera()
    n = int(args.seconds / model.opt.timestep)
    every = max(1, int(1.0 / (args.fps * model.opt.timestep)))
    frames = []
    for i in range(n):
        set_controls(model, data, i * model.opt.timestep, args.amp, args.freq,
                     args.outer_scale)
        mujoco.mj_step(model, data)
        if i % every == 0:
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    imageio.mimsave(args.record, frames, fps=args.fps, loop=0)
    print(f"wrote {args.record} ({len(frames)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default="urdf/gradus/sim/scene.xml")
    p.add_argument("--amp", type=float, default=220.0, help="peak tendon tension (N)")
    p.add_argument("--freq", type=float, default=0.35, help="circle frequency (Hz)")
    p.add_argument("--servo-kp", type=float, default=None,
                   help="override servo position gain (servo models only)")
    p.add_argument("--servo-kv", type=float, default=None,
                   help="override servo velocity gain (servo models only)")
    p.add_argument("--outer-scale", type=float, default=0.45,
                   help="outer tendon pull relative to inner (default 0.45)")
    p.add_argument("--record", default=None, help="write a GIF instead of opening a viewer")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()
    if args.record:
        record(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
