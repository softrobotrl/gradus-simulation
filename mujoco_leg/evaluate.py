"""Evaluate a trained tracer on held-out shapes; plot XY paths and render a GIF.

Example::

    python -m mujoco_leg.evaluate \
        --ckpt logs/mujoco_leg/learn2/best.pkl \
        --out logs/mujoco_leg/learn2/eval --gif
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import argparse
import dataclasses
import pickle
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mujoco_leg import shapes
from mujoco_leg.config import Config
from mujoco_leg.env import TracerEnv
from mujoco_leg.ppo import init_params, policy_mean


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    cfg = Config(**payload["config"])
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return cfg, params


def make_state(env: TracerEnv, shape, closed, key):
    st, _ = env.reset_one(key)
    st = st._replace(
        shape=shape,
        closed=jnp.asarray(closed),
        phase=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
        prev_action=jnp.zeros(env.nu),
    )
    return st._replace(prev_tip=st.data.site_xpos[env.tip_id])


def rollout(env: TracerEnv, params, apply, shape, closed, steps: int):
    st = make_state(env, shape, closed, jax.random.PRNGKey(1))

    def body(carry, _):
        st, = carry
        obs = env.obs(st)
        action = policy_mean(params, apply, obs[None])[0]
        target = shapes.target_at(st.shape, st.phase, st.closed, env.n_points)
        st, _, reward, done, info = env.step_one(st, action, auto_reset=False)
        tip = st.data.site_xpos[env.tip_id]
        return (st,), (tip[:2], target, st.data.qpos, st.data.qvel)

    (st,), traj = jax.lax.scan(body, (st,), None, length=steps)
    return traj


def plot_path(target_curve, actual, target, path: Path, name: str, rms: float, mx: float):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(target_curve[:, 0] * 1e3, target_curve[:, 1] * 1e3,
            "--", color="0.45", lw=1.5, label="target")
    ax.plot(actual[:, 0] * 1e3, actual[:, 1] * 1e3,
            "-", color="tab:blue", lw=1.6, label="tip")
    ax.scatter([actual[0, 0] * 1e3], [actual[0, 1] * 1e3],
               color="tab:green", s=20, zorder=5, label="start")
    ax.set_aspect("equal")
    lim = 55
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="0.9", lw=0.8)
    ax.axvline(0, color="0.9", lw=0.8)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{name}: RMS {rms*1e3:.2f} mm, max {mx*1e3:.2f} mm")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_gif(qpos: np.ndarray, qvel: np.ndarray, path: Path, every: int = 3, size: int = 340):
    model = mujoco.MjModel.from_xml_path("urdf/gradus/tracer/scene.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=size, width=size)

    persp = mujoco.MjvCamera()
    persp.type = mujoco.mjtCamera.mjCAMERA_FREE
    persp.lookat[:] = [0.0, 0.0, 0.30]
    persp.distance = 1.05
    persp.azimuth = 140.0
    persp.elevation = -18.0

    top = mujoco.MjvCamera()
    top.type = mujoco.mjtCamera.mjCAMERA_FREE
    top.lookat[:] = [0.0, 0.0, 0.60]
    top.distance = 0.32
    top.azimuth = 90.0
    top.elevation = -89.0

    frames = []
    for i in range(0, len(qpos), every):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=persp)
        left = renderer.render().copy()
        renderer.update_scene(data, camera=top)
        right = renderer.render().copy()
        frames.append(np.concatenate([left, right], axis=1))
    imageio.mimsave(path, frames, fps=25)
    renderer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()

    cfg, params = load_checkpoint(args.ckpt)
    out = args.out or args.ckpt.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    eval_cfg = dataclasses.replace(cfg, n_envs=1, warmup_steps=0)
    env = TracerEnv(eval_cfg)
    _, apply = init_params(jax.random.PRNGKey(0), env.obs_dim, env.nu, cfg.hidden)
    steps = args.steps or cfg.episode_steps

    named = shapes.eval_shapes(cfg.shape_radius, env.n_points)
    cases = [(k, v, True) for k, v in named.items()]
    for i in range(4):
        s, c = shapes.sample_shape(jax.random.PRNGKey(1000 + i), env.n_points, cfg.shape_radius)
        cases.append((f"random_{i}", s, bool(np.asarray(c))))

    print(f"{'shape':14s} {'RMS mm':>8s} {'max mm':>8s}")
    results = {}
    for name, shape, closed in cases:
        traj = rollout(env, params, apply, shape, closed, steps)
        tip, target, qpos, qvel = [np.asarray(x) for x in traj]
        skip = min(100, steps // 4)
        err = np.linalg.norm(tip[skip:] - target[skip:], axis=-1)
        rms = float(np.sqrt(np.mean(err ** 2)))
        mx = float(np.max(err))
        results[name] = (rms, mx)
        print(f"{name:14s} {rms*1e3:8.2f} {mx*1e3:8.2f}")

        if name in ("circle", "square", "star", "figure_eight", "random_0"):
            curve = np.asarray(shape)
            plot_path(curve, tip, target, out / f"eval_{name}.png", name, rms, mx)

        if args.gif and name == "circle":
            render_gif(qpos, qvel, out / "eval_circle.gif")

    with open(out / "metrics.txt", "w") as f:
        for name, (rms, mx) in results.items():
            f.write(f"{name} {rms:.8f} {mx:.8f}\n")
    print(f"* wrote {out}")


if __name__ == "__main__":
    main()
