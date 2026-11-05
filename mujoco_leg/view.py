"""Live MuJoCo viewer for the trained tracer.

Runs the policy on the CPU MuJoCo model (NumPy forward pass), so it does not
need JAX/CUDA and can share the desktop GL context with the viewer.

Example::

    DISPLAY=:0.0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
        python -m mujoco_leg.view --shape star

Blue points are the target curve; orange points are the tip trail.
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# checkpoint / policy
# --------------------------------------------------------------------------- #
def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["config"], payload["params"]


def policy_action(pi, obs: np.ndarray) -> np.ndarray:
    """Deterministic mean action of the stax MLP."""
    x = obs
    for layer in pi:
        if len(layer) == 0:          # Tanh
            x = np.tanh(x)
        else:
            w, b = layer
            x = x @ w + b
    return np.tanh(x[: x.shape[0] // 2])


# --------------------------------------------------------------------------- #
# shapes (numpy versions of mujoco_leg.shapes)
# --------------------------------------------------------------------------- #
def make_shape(name: str, n_points: int, radius: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_points, endpoint=False)
    if name == "circle":
        return np.stack([np.cos(TWO_PI * t), np.sin(TWO_PI * t)], axis=-1) * radius
    if name == "square":
        t4 = t * 4.0
        x = np.where(t4 < 1.0, t4, np.where(t4 < 2.0, 1.0, np.where(t4 < 3.0, 3.0 - t4, 0.0)))
        y = np.where(t4 < 1.0, 0.0, np.where(t4 < 2.0, t4 - 1.0, np.where(t4 < 3.0, 1.0, 4.0 - t4)))
        return (np.stack([x, y], axis=-1) - 0.5) * (2.0 * radius)
    if name == "star":
        r = np.where((t * 10.0).astype(np.int32) % 2 == 0, 1.0, 0.45) * radius
        a = TWO_PI * t - np.pi / 2.0
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)
    if name == "figure_eight":
        xy = np.stack([np.sin(TWO_PI * t), 0.5 * np.sin(2.0 * TWO_PI * t)], axis=-1)
        xy -= xy.mean(axis=0, keepdims=True)
        return xy * (radius / (np.max(np.linalg.norm(xy, axis=-1)) + 1e-9))
    raise ValueError(name)


def phase_to_u(phase: float, closed: bool) -> float:
    return phase if closed else 1.0 - abs(2.0 * phase - 1.0)


def target_at(shape: np.ndarray, phase: float, closed: bool, n_points: int) -> np.ndarray:
    u = phase_to_u(phase, closed) % 1.0
    idx = u * n_points
    i0 = int(np.floor(idx)) % n_points
    i1 = (i0 + 1) % n_points
    frac = idx - np.floor(idx)
    return shape[i0] + frac * (shape[i1] - shape[i0])


def target_window(shape, phase, closed, lookahead, lookahead_dt, period, n_points):
    offs = np.arange(lookahead + 1) * (lookahead_dt / period)
    phases = (phase + offs) % 1.0
    return np.stack([target_at(shape, ph, closed, n_points) for ph in phases])


# --------------------------------------------------------------------------- #
# observation (must match mujoco_leg.env.TracerEnv.obs)
# --------------------------------------------------------------------------- #
def observation(m, d, shape, phase, prev_tip, l0, stroke, fmax, tip_id, cfg):
    tip = d.site_xpos[tip_id].copy()
    tip_vel = (tip - prev_tip) / (m.opt.timestep * cfg["n_substeps"])
    window = target_window(
        shape, phase, True,
        cfg["lookahead"], cfg["lookahead_dt"], cfg["period"], cfg["shape_points"],
    )
    return np.concatenate([
        d.qpos,
        d.qvel / 10.0,
        (d.ten_length - l0) / stroke,
        -d.actuator_force / fmax,
        tip[:2] / 0.05,
        tip_vel[:2] / 0.2,
        window.reshape(-1) / 0.05,
        np.array([np.sin(TWO_PI * phase), np.cos(TWO_PI * phase)]),
    ])


# --------------------------------------------------------------------------- #
# viewer
# --------------------------------------------------------------------------- #
def add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path,
                   default=Path("logs/mujoco_leg/tracer_final/final.pkl"))
    p.add_argument("--shape", default="star",
                   choices=["star", "circle", "square", "figure_eight"])
    p.add_argument("--view", default="top", choices=["top", "perspective"])
    p.add_argument("--radius", type=float, default=None,
                   help="shape radius in metres (default: training radius)")
    p.add_argument("--trail", type=int, default=400)
    args = p.parse_args()

    cfg, params = load_checkpoint(args.ckpt)
    pi = params["pi"]
    radius = args.radius if args.radius is not None else cfg["shape_radius"]
    shape = make_shape(args.shape, cfg["shape_points"], radius)

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

    trail = deque(maxlen=args.trail)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if args.view == "top":
            viewer.cam.lookat[:] = [0.0, 0.0, tip_z]
            viewer.cam.distance = 0.40
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -89.0
        else:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.50]
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = 20.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1

        last = time.time()
        while viewer.is_running():
            obs = observation(model, data, shape, phase, prev_tip, l0, stroke, fmax, tip_id, cfg)
            action = policy_action(pi, obs)

            pull = 0.5 * (action + 1.0)
            target_ctrl = l0 - stroke * pull
            ctrl = 0.5 * data.ctrl + 0.5 * target_ctrl
            data.ctrl[:] = np.clip(ctrl, l0 - stroke, l0)

            for _ in range(cfg["n_substeps"]):
                mujoco.mj_step(model, data)

            tip = data.site_xpos[tip_id].copy()
            phase = (phase + dt / cfg["period"]) % 1.0
            prev_tip = tip
            trail.append(tip.copy())

            scn = viewer.user_scn
            scn.ngeom = 0
            for q in shape[::2]:
                add_sphere(scn, (q[0], q[1], z_draw), 0.0012, (0.2, 0.45, 1.0, 0.55))
            for q in trail:
                add_sphere(scn, (q[0], q[1], q[2] + 0.02), 0.0008, (1.0, 0.55, 0.1, 0.85))
            viewer.sync()

            elapsed = time.time() - last
            if elapsed < dt:
                time.sleep(dt - elapsed)
            last = time.time()


if __name__ == "__main__":
    main()
