"""Small checks for the single demo leg rig; the env test is skipped until built."""

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from mujoco_to_mjcf.merge import _dir_of, _slug
from single_leg_demo.rig_env import SingleLegRigEnv, RigConfig, _bend_angle, _quat_to_rotvec

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
            self.assertEqual(obs["policy"].shape, (2, env.num_obs))
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


if __name__ == "__main__":
    unittest.main()
