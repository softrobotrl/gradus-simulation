"""Quick checks for the sliding-rig env.

Run after building the rig::

    python -m single_leg_demo.build_rig
    python -m single_leg_demo.validate --num-envs 4
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from single_leg_demo.rig_env import N_BALL, N_TENDON, RigConfig, SingleLegRigEnv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()

    cfg = RigConfig(num_envs=args.num_envs, num_threads=args.num_envs,
                    episode_steps=args.steps + 10, reset_noise=0.0)
    env = SingleLegRigEnv(cfg)
    print(f"* model: nq={env.nq} nv={env.nv} nu={env.nu} obs={env.num_obs} "
          f"actions={env.num_actions}")
    assert env.num_obs == 115, env.num_obs
    assert env.nu == N_TENDON
    assert len(env.ball_qadr) == N_BALL

    obs = env.reset()
    assert torch.isfinite(obs["policy"]).all()

    # Zero action must hold the reset pose.
    x0 = np.array([d.qpos[0] for d in env.datas])
    obs, _, _, _ = env.step(torch.zeros(env.num_envs, N_TENDON))
    assert torch.isfinite(obs["policy"]).all()
    drift = np.abs(np.array([d.qpos[0] for d in env.datas]) - x0)
    print(f"* zero-action drift over one control step: {drift.max():.5f} m")
    assert drift.max() < 0.02, drift

    # A scripted pull must be able to move the carriage both ways.
    for sign, label in ((1.0, "+X"), (-1.0, "-X")):
        obs = env.reset()
        for i in range(env.num_envs):
            env.cmd_mag[i] = 0.10
            env.cmd_sign0[i] = sign
            env.cmd_period[i] = 1e6
            env._set_cmd(i)
        start = np.array([d.qpos[0] for d in env.datas])
        # +X motion needs the foot to move -X, i.e. pull the back tendons.
        action = torch.zeros(env.num_envs, N_TENDON)
        action[:, 2] = 1.0
        action[:, 6] = 1.0
        if sign < 0:
            action.zero_()
            action[:, 0] = 1.0
            action[:, 4] = 1.0
        for _ in range(args.steps):
            obs, _, _, _ = env.step(action)
        dx = np.array([d.qpos[0] for d in env.datas]) - start
        print(f"* scripted {label} pull: dx mean={dx.mean():+.4f} m "
              f"range=[{dx.min():+.4f}, {dx.max():+.4f}]")
        assert np.all(dx * sign > -0.01), dx

    env.close()
    print("* ok")


if __name__ == "__main__":
    main()
