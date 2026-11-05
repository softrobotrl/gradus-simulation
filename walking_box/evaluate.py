"""Evaluate a trained walking-box policy and render a GIF.

Example::

    python -m walking_box.evaluate \
        --ckpt logs/walking_box/walk1/model_1500.pt --cmd 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from walking_box.cpu_env import EnvConfig, QuadrupedCpuEnv
from walking_box.train_rsl import training_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("logs/walking_box/walk1/model_1500.pt"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cmd", type=float, default=0.2)
    p.add_argument("--gait-amp", type=float, default=None)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--size", type=int, default=480)
    args = p.parse_args()

    out = args.out or args.ckpt.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(
        num_envs=1,
        num_threads=1,
        episode_steps=args.steps,
        gait_amp=0.5 if args.gait_amp is None else args.gait_amp,
    )
    env = QuadrupedCpuEnv(cfg)
    runner = OnPolicyRunner(env, training_config("eval", 24), None, device="cpu")
    runner.load(str(args.ckpt), map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    obs = env.reset()
    env.cmd[0, 0] = args.cmd
    env.cmd[0, 1:] = 0.0
    env.obs[0] = env._obs_one(0)
    obs = env.get_observations()

    qpos, qvel, vxs, zs = [], [], [], []
    for _ in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, reward, done, extras = env.step(actions, auto_reset=False)
        qpos.append(env.datas[0].qpos.copy())
        qvel.append(env.datas[0].qvel.copy())
        vxs.append(float(extras["log"]["/vx_mean"]))
        zs.append(float(extras["log"]["/height_mean"]))
        if bool(done[0]):
            break
    env.close()

    qpos = np.asarray(qpos)
    qvel = np.asarray(qvel)
    vx = float(np.mean(vxs)) if vxs else 0.0
    z = float(np.mean(zs)) if zs else 0.0
    dx = float(qpos[-1, 0] - qpos[0, 0]) if len(qpos) else 0.0
    print(f"* steps={len(qpos)}  mean_vx={vx:.3f} m/s  dx={dx:.3f} m  z={z:.3f} m")
    (out / "metrics.txt").write_text(f"mean_vx {vx:.6f}\ndx {dx:.6f}\nz {z:.6f}\n")

    # render the full visual model
    model = mujoco.MjModel.from_xml_path("urdf/gradus/walk/quadruped.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.6
    cam.azimuth = 135.0
    cam.elevation = -18.0

    frames = []
    for i in range(0, len(qpos), args.every):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        # follow the trunk so the gait stays framed as the robot moves
        cam.lookat[:] = data.qpos[:3]
        cam.lookat[2] -= 0.05
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    gif = out / "walk.gif"
    imageio.mimsave(gif, frames, fps=25)
    print(f"* wrote {gif}")


if __name__ == "__main__":
    main()
