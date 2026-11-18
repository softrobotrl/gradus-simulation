"""Evaluate a trained sliding-rig policy and render a GIF.

Example::

    python -m single_leg_demo.evaluate \
        --ckpt logs/single_leg_demo/rig1/model_500.pt --steps 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from single_leg_demo.rig_env import FOOT_RADIUS, RigConfig, SingleLegRigEnv
from single_leg_demo.train import training_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--cmd-min", type=float, default=0.10)
    p.add_argument("--cmd-max", type=float, default=0.10)
    p.add_argument("--cmd-period", type=float, default=2.0)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    out = args.out or args.ckpt.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)

    # Use the same fixed command normaliser that the checkpoint was trained
    # with.  Older runs did not save obs_cmd_scale, so fall back to cmd_max.
    # Runs from before the hardware observation have no obs_mode: they used
    # the full 115-value state.
    obs_scale = 0.16
    # Runs from before the servo speed limit / fixed pull_bias used neither.
    obs_mode, obs_history = "full", 1
    servo_speed, fixed_pull_bias = 0.0, False
    critic_dr_params = False          # older critics had no randomized-value inputs
    train_cfg = args.ckpt.parent / "config.json"
    if train_cfg.exists():
        saved = json.loads(train_cfg.read_text()).get("environment", {})
        obs_scale = float(saved.get("obs_cmd_scale", saved.get("cmd_max", obs_scale)))
        obs_mode = saved.get("obs_mode", obs_mode)
        obs_history = int(saved.get("obs_history", obs_history))
        servo_speed = float(saved.get("servo_speed", servo_speed))
        fixed_pull_bias = bool(saved.get("fixed_pull_bias", fixed_pull_bias))
        critic_dr_params = bool(saved.get("critic_dr_params", critic_dr_params))

    cfg = RigConfig(
        num_envs=1,
        num_threads=1,
        episode_steps=args.steps,
        cmd_min=args.cmd_min,
        cmd_max=args.cmd_max,
        cmd_period=args.cmd_period,
        cmd_period_jitter=0.0,
        cmd_resample=False,
        obs_cmd_scale=obs_scale,
        obs_mode=obs_mode,
        obs_history=obs_history,
        servo_speed=servo_speed,
        fixed_pull_bias=fixed_pull_bias,
        critic_dr_params=critic_dr_params,
        reset_noise=0.0,
    )
    env = SingleLegRigEnv(cfg)
    runner = OnPolicyRunner(env, training_config("eval", 24), None, device="cpu")
    runner.load(str(args.ckpt), map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    obs = env.reset()
    env.cmd_mag[0] = args.cmd_max
    env.cmd_sign0[0] = 1.0
    env.cmd_period[0] = args.cmd_period
    env._set_cmd(0)
    env._push_obs(0, fresh=True)
    obs = env.get_observations()

    qpos, qvel, xs, vxs, cmds, foot_zs, bends = [], [], [], [], [], [], []
    failed = False
    for _ in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, reward, done, extras = env.step(actions, auto_reset=False)
        d = env.datas[0]
        qpos.append(d.qpos.copy())
        qvel.append(d.qvel.copy())
        xs.append(float(d.qpos[0]))
        vxs.append(float(d.qvel[0]))
        cmds.append(float(env.cmd[0]))
        foot_zs.append(float(d.site_xpos[env.tip_id][2]))
        bends.append(float(extras["log"]["/bend_max"]))
        if bool(done[0]):
            failed = not bool(extras["time_outs"][0])
            break
    dt = env.dt
    model = env.m
    env.close()

    qpos = np.asarray(qpos)
    qvel = np.asarray(qvel)
    xs = np.asarray(xs)
    vxs = np.asarray(vxs)
    cmds = np.asarray(cmds)
    foot_zs = np.asarray(foot_zs)
    bends = np.asarray(bends)
    err = vxs - cmds
    duration = len(xs) * dt
    contact = float(np.mean(foot_zs <= FOOT_RADIUS + 0.01)) if len(foot_zs) else 0.0
    reversals = int(np.sum(np.diff(np.sign(cmds)) != 0)) if len(cmds) else 0
    print(
        f"* steps={len(xs)}  mean|err|={np.mean(np.abs(err)):.3f} m/s  "
        f"rmse={np.sqrt(np.mean(err**2)):.3f}  mean_vx={vxs.mean():+.3f} m/s  "
        f"x=[{xs.min():+.3f}, {xs.max():+.3f}] m  span={xs.max()-xs.min():.3f} m  "
        f"contacts={contact:.2f}  bend_max={bends.max():.2f} rad  "
        f"reversals={reversals}  failed={failed}"
    )
    (out / "metrics.txt").write_text(
        f"steps {len(xs)}\nduration {duration:.3f}\n"
        f"mean_abs_err {np.mean(np.abs(err)):.6f}\nrmse {np.sqrt(np.mean(err**2)):.6f}\n"
        f"mean_vx {vxs.mean():.6f}\nx_min {xs.min():.6f}\nx_max {xs.max():.6f}\n"
        f"span {xs.max()-xs.min():.6f}\ncontact_frac {contact:.4f}\n"
        f"bend_max {bends.max():.6f}\nreversals {reversals}\nfailed {int(failed)}\n"
    )
    if args.no_render:
        return

    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.05
    cam.azimuth = 135.0
    cam.elevation = -12.0

    data = mujoco.MjData(model)
    frames = []
    for i in range(0, len(qpos), args.every):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [float(data.qpos[0]), 0.0, 0.30]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    gif = out / "rig.gif"
    fps = 1.0 / (dt * args.every)
    imageio.mimsave(gif, frames, fps=fps)
    print(f"* wrote {gif} ({len(frames)} frames at {fps:.1f} fps)")


if __name__ == "__main__":
    main()
