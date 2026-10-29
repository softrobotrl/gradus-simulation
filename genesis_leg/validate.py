"""Check that tendon actuation and episode resets work before training."""

import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 10:
        parser.error("Require num-envs >= 2 and steps >= 10")
    from genesis_leg.env import GenesisLegEnv, LegConfig
    from genesis_leg.train import initialize

    initialize(args.backend, 0)
    env = GenesisLegEnv(LegConfig(num_envs=args.num_envs))
    bends = []
    for tendon in range(env.num_actions):
        env.reset()
        initial_tip = env.robot.get_links_pos()[:, env.tip_link, :2].clone()
        action = torch.full_like(env.targets, -0.15)
        action[:, tendon] = 0.1
        for _ in range(args.steps):
            obs, reward, done, extras = env.step(action)
            assert torch.isfinite(obs["policy"]).all(), "Nonfinite observations"
            assert torch.isfinite(reward).all(), "Nonfinite rewards"
            assert not done.any(), f"Episode ended during tendon {tendon} pull"
        displacement = env.robot.get_links_pos()[:, env.tip_link, :2] - initial_tip
        direction = env.offsets[tendon, 2, :2]
        direction = direction / torch.linalg.vector_norm(direction)
        bend_m = displacement @ direction
        assert (bend_m > 0.001).all(), f"Tendon {tendon} did not bend toward its guide"
        bends.append(bend_m.mean().item() * 1000)

    # Time out one environment and check that the others keep running.
    env.reset()
    env.episode_length_buf[0] = env.max_episode_length - 1
    _, _, done, extras = env.step(torch.zeros_like(env.targets))
    assert done[0] and extras["time_outs"][0] == 1, "Timeout classification failed"
    assert not done[1:].any(), "Reset leaked across environments"
    assert env.episode_length_buf[0] == 0, "Episode clock did not reset"
    assert (env.episode_length_buf[1:] == 1).all(), "Other episode clocks were reset"
    assert extras["final_observation"].shape == env.obs_buf.shape
    results = {
        "status": "PASS",
        "num_envs": env.num_envs,
        "observations": env.num_obs,
        "actions": env.num_actions,
        "pull_bend_mm": bends,
        "checks": ["finite stepping", "tendon direction", "timeout/reset isolation"],
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
