"""Evaluate or watch a trained SAC policy on the board-pushing task.

Usage:
  python -m policy.play_sac logs/sac/<run>/checkpoints/best.pt                  # evaluate 64 episodes, print stats
  python -m policy.play_sac logs/sac/<run>/checkpoints/best.pt --view           # watch it live in the MuJoCo viewer
  python -m policy.play_sac ... --episodes 16 --stochastic                      # sample actions instead of the mean

The evaluation uses the deterministic policy (the mean action), which is how a policy runs on hardware.
"""

import argparse
import time

import numpy as np
import torch

from pipeline.board_env import BoardPushEnvCPU, make_board_env
from policy.sac import SAC
from policy.train_sac import env_kwargs, load_config, pick_learner


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
    env = make_board_env(cfg.config, episodes, backend=backend, seed=12345, **env_kwargs(cfg))
    agent = build_agent(ckpt, cfg, env, device)
    layout, dt = env.obs_layout, env.jax_env.dt
    obs = env.reset()["policy"]
    direction = np.array(env.state.info["direction"])
    alive = np.ones(episodes, bool)  # still in its first episode
    along, across = np.zeros(episodes), np.zeros(episodes)  # board displacement, summed per step
    length, ret, truncated = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes, bool)
    max_lift, min_gap, max_tension = 0.0, 0.0, 0.0
    for t in range(env.max_episode_length):
        prev_xy = np.array(env.state.info["board_xy"])
        action = agent.act(obs, deterministic).to(env.device)
        next_obs, reward, done, extras = env.step(action)
        done = done.cpu().numpy().astype(bool)
        final = extras["final_observation"].cpu().numpy()
        # Envs that just finished were already reset, so their last move is estimated from the board velocity.
        moved = np.where(done[:, None], final[:, layout["board_vel"]][:, :2] * dt,
                         np.array(env.state.info["board_xy"]) - prev_xy)
        live = alive
        along[live] += (moved * direction).sum(1)[live]
        across[live] += (moved[:, 1] * direction[:, 0] - moved[:, 0] * direction[:, 1])[live]
        ret[live] += reward.cpu().numpy()[live]
        length[live] += 1
        truncated |= live & done & (t + 1 < env.max_episode_length)
        max_lift = max(max_lift, float(final[live][:, layout["board_lift"]].max(initial=0)))
        min_gap = min(min_gap, float(final[live][:, layout["pad_gap"]].min(initial=0)))
        max_tension = max(max_tension, float(np.abs(final[live][:, layout["tendon_tension"]]).max(initial=0)))
        alive = alive & ~done
        obs = next_obs["policy"]
        if not alive.any():
            break
    travel, lateral = along, np.abs(across)
    seconds = length * env.jax_env.dt
    print(f"\n{episodes} episodes, {'deterministic' if deterministic else 'stochastic'} policy, "
          f"checkpoint step {ckpt['step']:,}")
    print(f"  board speed along direction  {np.mean(travel / seconds) * 1e3:6.2f} mm/s  "
          f"(min {np.min(travel / seconds) * 1e3:.2f}, max {np.max(travel / seconds) * 1e3:.2f})")
    print(f"  travel per episode           {np.mean(travel) * 1e3:6.1f} mm in {np.mean(seconds):.1f} s on average")
    print(f"  sideways drift per episode   {np.mean(lateral) * 1e3:6.1f} mm")
    print(f"  episode return               {np.mean(ret):6.1f}")
    print(f"  ended early (ran out of board) {truncated.mean():.0%}")
    print(f"  physics sanity: max board lift {max_lift * 1e3:.1f} mm, deepest pad penetration "
          f"{-min_gap * 1e3:.2f} mm, max tendon tension {max_tension:.1f} N")


def view(path: str, deterministic: bool) -> None:
    import mujoco.viewer

    device = pick_learner("auto")
    ckpt, cfg = load_agent(path, device)
    env = BoardPushEnvCPU(cfg.config, 1, **env_kwargs(cfg))
    agent = build_agent(ckpt, cfg, env, device)
    obs = env.reset()["policy"]
    dt = env.jax_env.dt
    print("viewer open: close the window to stop. Episodes restart automatically.")
    with mujoco.viewer.launch_passive(env.mj_model, env.datas[0]) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            obs, reward, done, extras = env.step(agent.act(obs, deterministic).cpu())
            obs = obs["policy"]
            viewer.sync()
            if done.any():
                print("episode ended (time limit or ran out of board); restarting")
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))  # real time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer instead of evaluating")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    parser.add_argument("--backend", default="auto", choices=["auto", "mjx", "cpu"])
    args = parser.parse_args()
    if args.view:
        view(args.checkpoint, not args.stochastic)
    else:
        evaluate(args.checkpoint, args.episodes, not args.stochastic, args.backend)
