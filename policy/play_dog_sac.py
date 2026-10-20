"""Evaluate or watch a trained SAC policy on the Gradus dog.

Usage::

    python -m policy.play_dog_sac logs/dog_sac/<run>/checkpoints/best.pt
    python -m policy.play_dog_sac logs/dog_sac/<run>/checkpoints/best.pt --view
    python -m policy.play_dog_sac ... --episodes 16 --stochastic

The evaluation uses the deterministic policy (the mean action), which is how a
policy would run on hardware.  Each of the ``--episodes`` environments starts
one episode; a fallen environment is not counted again after its first reset,
so the printed numbers describe the first episode of every environment.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from pipeline.dog_env import DogEnv, DogEnvConfig, make_dog_env
from policy.sac import SAC
from policy.train_dog_sac import env_kwargs, load_config, pick_learner


def load_agent(path: str, device: str):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    cfg = load_config(ckpt["config"])
    return ckpt, cfg


def build_agent(ckpt, cfg, env, device) -> SAC:
    agent = SAC(env.num_obs, env.num_actions, cfg.sac, device)
    agent.load_state_dict(ckpt["agent"])
    return agent


def evaluate(path: str, episodes: int, deterministic: bool, backend: str) -> None:
    device = pick_learner("auto")
    ckpt, cfg = load_agent(path, device)
    env = make_dog_env(episodes, backend=backend, seed=12345, **env_kwargs(cfg))
    agent = build_agent(ckpt, cfg, env, device)
    layout = env.obs_layout
    obs = env.reset()["policy"]

    alive = np.ones(episodes, dtype=bool)
    ret = np.zeros(episodes)
    length = np.zeros(episodes, dtype=np.int64)
    failed = np.zeros(episodes, dtype=bool)
    speed_sum = np.zeros(episodes)
    vel_err_sum = np.zeros(episodes)
    tension_sum = np.zeros(episodes)
    contact_sum = np.zeros(episodes)
    height_end = np.zeros(episodes)
    tilt_end = np.zeros(episodes)

    for _ in range(env.max_episode_length):
        action = agent.act(obs, deterministic).to(env.device)
        next_obs, reward, done, extras = env.step(action)
        r = reward.cpu().numpy()
        d = done.cpu().numpy().astype(bool)
        time_out = extras["time_outs"].cpu().numpy().astype(bool)
        final = extras["final_observation"].cpu().numpy()

        height = final[:, layout["base_height_residual"]].reshape(-1) + cfg.height_target
        gravity_residual = final[:, layout["projected_gravity_residual"]]
        tilt = np.linalg.norm(gravity_residual[:, :2], axis=1)
        lin_vel = final[:, layout["base_lin_vel"]]
        speed = np.linalg.norm(lin_vel[:, :2], axis=1)
        vel_residual = final[:, layout["velocity_residual"]]
        vel_err = np.linalg.norm(vel_residual[:, :2], axis=1)
        tension = final[:, layout["tendon_tension"]].mean(axis=1)
        contact = final[:, layout["foot_contact"]].mean(axis=1)

        live = alive.copy()
        ret[live] += r[live]
        length[live] += 1
        speed_sum[live] += speed[live]
        vel_err_sum[live] += vel_err[live]
        tension_sum[live] += tension[live]
        contact_sum[live] += contact[live]
        height_end[live] = height[live]
        tilt_end[live] = tilt[live]
        failed[live & d & ~time_out] = True
        alive &= ~d
        obs = next_obs["policy"]
        if not alive.any():
            break

    steps = np.maximum(length, 1)
    seconds = length * env.cfg.control_dt
    print(f"\n{episodes} episodes, {'deterministic' if deterministic else 'stochastic'} policy, "
          f"checkpoint step {ckpt['step']:,}")
    print(f"  episode return              {np.mean(ret):7.2f}  (min {np.min(ret):.2f}, max {np.max(ret):.2f})")
    print(f"  episode length              {np.mean(seconds):7.2f} s  (failure rate {failed.mean():.0%})")
    print(f"  mean speed                  {np.mean(speed_sum / steps):7.2f} m/s")
    print(f"  command tracking error      {np.mean(vel_err_sum / steps):7.2f} m/s")
    print(f"  final base height           {np.mean(height_end):7.3f} m  (target {cfg.height_target:.3f})")
    print(f"  final tilt                  {np.mean(tilt_end):7.3f}")
    print(f"  mean normalized tension     {np.mean(tension_sum / steps):7.3f}")
    print(f"  mean foot contact           {np.mean(contact_sum / steps):7.2f} / 4")
    print(f"  log: {path}")
    env.close()


def view(path: str, deterministic: bool) -> None:
    import mujoco.viewer

    device = pick_learner("auto")
    ckpt, cfg = load_agent(path, device)
    kwargs = env_kwargs(cfg)
    kwargs["num_threads"] = 1
    env = DogEnv(1, DogEnvConfig(**kwargs), seed=0)
    agent = build_agent(ckpt, cfg, env, device)
    obs = env.reset()["policy"]
    print("viewer open: close the window to stop. Episodes restart automatically.")
    with mujoco.viewer.launch_passive(env.mj_model, env.datas[0]) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.18]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        while viewer.is_running():
            t0 = time.perf_counter()
            action = agent.act(obs, deterministic).cpu()
            next_obs, reward, done, extras = env.step(action)
            obs = next_obs["policy"]
            viewer.sync()
            if done.any():
                print("episode ended; reset")
            time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer instead of evaluating")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "auto", "mjx"])
    args = parser.parse_args()
    if args.view:
        view(args.checkpoint, not args.stochastic)
    else:
        evaluate(args.checkpoint, args.episodes, not args.stochastic, args.backend)
