"""Small checks for the single demo leg rig; the env test is skipped until built."""

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from mujoco_to_mjcf.merge import _dir_of, _slug
from single_leg_demo.rig_env import (
    HW_FRAME, N_TENDON, SingleLegRigEnv, RigConfig, _bend_angle, _quat_to_rotvec)

RIG_XML = Path("urdf/gradus/demo/rig.xml")


class GeometryTests(unittest.TestCase):
    def test_slug_matches_onshape_to_robot(self):
        self.assertEqual(_slug("Shin V4 <1>"), "shin_v4")
        self.assertEqual(_slug("Shin V4 <2>"), "shin_v4_2")
        self.assertEqual(_slug("Thigh Segment 2H v4 <4>"), "thigh_segment_2h_v4_4")
        self.assertEqual(_slug("ASY-HIP-MOUNT <1>"), "asy_hip_mount")
        self.assertEqual(_slug("ASY-FOOT-ADAPTOR <1>"), "asy_foot_adaptor")

    def test_forty_five_degree_station_still_has_four_directions(self):
        yaw = math.radians(-44.5)
        for offset in (0.0, math.pi / 4):
            dirs = {
                _dir_of(offset + k * math.pi / 2, yaw) for k in range(4)
            }
            self.assertEqual(dirs, {"front", "left", "back", "right"})

    def test_quat_helpers(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(_quat_to_rotvec(identity), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(_bend_angle(identity[None])[0]), 0.0)
        q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
        np.testing.assert_allclose(_quat_to_rotvec(q), [0.0, 0.0, math.pi / 2], atol=1e-6)
        self.assertAlmostEqual(float(_bend_angle(q[None])[0]), math.pi / 2, places=6)


@unittest.skipUnless(RIG_XML.exists(), "run `python -m single_leg_demo.build_rig` first")
class RigEnvTests(unittest.TestCase):
    def test_observation_and_actuation(self):
        cfg = RigConfig(num_envs=2, num_threads=2, episode_steps=20,
                        reset_noise=0.0, cmd_min=0.1, cmd_max=0.1)
        env = SingleLegRigEnv(cfg)
        try:
            obs = env.reset()
            self.assertEqual(obs["policy"].shape, (2, HW_FRAME * cfg.obs_history))
            self.assertEqual(obs["critic"].shape, (2, 115))
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            start = np.array([d.qpos[0] for d in env.datas])
            action = torch.zeros(2, env.num_actions)
            action[:, [2, 6]] = 1.0
            for _ in range(80):
                obs, reward, done, extras = env.step(action, auto_reset=False)
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            self.assertTrue(torch.isfinite(reward).all())
            dx = np.array([d.qpos[0] for d in env.datas]) - start
            self.assertTrue(np.all(dx > 0.0), dx)
        finally:
            env.close()

    def test_hardware_frame_layout_and_history(self):
        cfg = RigConfig(num_envs=1, num_threads=1, episode_steps=20, reset_noise=0.0,
                        cmd_min=0.1, cmd_max=0.1, obs_history=3)
        env = SingleLegRigEnv(cfg)
        try:
            env.reset()
            hist = env.hist[0]
            # fresh reset fills every history slot with the same step
            np.testing.assert_allclose(hist[0], hist[-1])
            frame = hist[-1]
            n = N_TENDON
            prev_action = frame[3:3 + n]
            target = frame[3 + n:3 + 2 * n]
            position = frame[3 + 2 * n:3 + 3 * n]
            load = frame[3 + 3 * n:3 + 4 * n]
            acc = frame[3 + 4 * n:3 + 4 * n + 3]
            np.testing.assert_allclose(prev_action, 0.0)
            self.assertTrue(np.all((target >= 0) & (target <= 1)))
            # the spool cannot be pulled further than it was told to go
            self.assertTrue(np.all(position <= target + 1e-9))
            self.assertTrue(np.all(load >= 0))
            self.assertAlmostEqual(float(acc[2]), 1.0, delta=0.05)

            action = torch.full((1, env.num_actions), 0.5)
            obs, *_ = env.step(action, auto_reset=False)
            hist = env.hist[0]
            # history shifts: newest step holds the action just applied
            np.testing.assert_allclose(hist[-1, 3:3 + n], 0.5)
            np.testing.assert_allclose(hist[-2, 3:3 + n], 0.0)
            np.testing.assert_allclose(obs["policy"][0].numpy(), hist.reshape(-1), rtol=1e-6)
        finally:
            env.close()

    def test_full_mode_matches_legacy_size(self):
        cfg = RigConfig(num_envs=1, num_threads=1, episode_steps=20, reset_noise=0.0,
                        obs_mode="full")
        env = SingleLegRigEnv(cfg)
        try:
            obs = env.reset()
            self.assertEqual(obs["policy"].shape, (1, 115))
            torch.testing.assert_close(obs["policy"], obs["critic"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
