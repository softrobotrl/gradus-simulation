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


def saved_env_settings(ckpt: Path) -> dict:
    """RigConfig settings a checkpoint was trained with, from its run's config.json.

    Older runs miss newer keys, so each falls back to how those runs worked:
    no obs_cmd_scale -> cmd_max; no obs_mode -> full 115-value state; no servo
    speed limit or fixed pull_bias; no randomized values or scored speed in the
    critic; no cmd_ramp -> step command (before reward v2); no rail_friction ->
    the value in rig.xml; no servo_load_obs -> loads in the actor input (before
    the PWM servos were confirmed); no encoder_counts_per_rev -> no rounding;
    no servo_layout -> 8 independent servos at rig.xml's force limit; no
    imu_obs -> rail IMU in the actor input (before the no-IMU rig, 2026-11-26).
    """
    settings = {
        "obs_cmd_scale": 0.16, "obs_mode": "full", "obs_history": 1,
        "servo_speed": 0.0, "fixed_pull_bias": False, "critic_dr_params": False,
        "critic_scored_speed": False, "cmd_ramp": 0.0, "rail_friction": -1.0,
        "servo_load_obs": True, "encoder_counts_per_rev": 0, "spool_radius": 0.010,
        "servo_layout": "independent", "servo_torque": 0.0, "imu_obs": True, "leg_yaw_deg": 0.0,
    }
    train_cfg = ckpt.parent / "config.json"
    if train_cfg.exists():
        saved = json.loads(train_cfg.read_text()).get("environment", {})
        settings["obs_cmd_scale"] = float(saved.get("obs_cmd_scale",
                                                    saved.get("cmd_max", 0.16)))
        for key in ("obs_mode", "obs_history", "servo_speed", "fixed_pull_bias",
                    "critic_dr_params", "critic_scored_speed", "cmd_ramp", "rail_friction",
                    "servo_load_obs", "encoder_counts_per_rev", "spool_radius",
                    "servo_layout", "servo_torque", "imu_obs", "leg_yaw_deg"):
            if key in saved:
                settings[key] = type(settings[key])(saved[key])
    return settings


class NumpyPolicy:
    """A policy exported by train_mjx.py (policy.npz): input normalizer + MLP,
    deterministic (the mean action).  No JAX needed."""

    def __init__(self, path: Path, num_actions: int):
        z = np.load(path)
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        n = len([k for k in z.files if k.startswith("w")])
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]
        self.act = str(z["activation"])
        self.n = num_actions

    def __call__(self, obs):
        x = obs["policy"].numpy() if hasattr(obs, "keys") else np.asarray(obs)
        x = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0))) if self.act == "elu" else np.maximum(x, 0)
        return torch.from_numpy(x[..., :self.n].astype(np.float32))


def load_policy(env: SingleLegRigEnv, ckpt: Path):
    """Inference policy from a checkpoint (rsl-rl .pt, or a train_mjx policy.npz);
    works with any env of the same shape."""
    if Path(ckpt).suffix == ".npz":
        return NumpyPolicy(Path(ckpt), env.num_actions)
    runner = OnPolicyRunner(env, training_config("eval", 24), None, device="cpu")
    runner.load(str(ckpt), map_location="cpu")
    return runner.get_inference_policy(device="cpu")


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

    cfg = RigConfig(
        num_envs=1,
        num_threads=1,
        episode_steps=args.steps,
        cmd_min=args.cmd_min,
        cmd_max=args.cmd_max,
        cmd_period=args.cmd_period,
        cmd_period_jitter=0.0,
        cmd_resample=False,
        reset_noise=0.0,
        **saved_env_settings(args.ckpt),
    )
    env = SingleLegRigEnv(cfg)
    policy = load_policy(env, args.ckpt)

    obs = env.reset()
    env.cmd_mag[0] = args.cmd_max
    env.cmd_sign0[0] = 1.0
    env.cmd_period[0] = args.cmd_period
    env.cmd[0] = 0.0  # restart a ramped command from rest
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
