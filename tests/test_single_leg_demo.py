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
            self.assertEqual(obs["critic"].shape, (2, 115 + 12))
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

    def test_servo_speed_limit(self):
        cfg = RigConfig(num_envs=1, num_threads=1, episode_steps=20, reset_noise=0.0)
        env = SingleLegRigEnv(cfg)
        try:
            env.reset()
            before = env.datas[0].ctrl.copy()
            env.step(torch.ones(1, env.num_actions), auto_reset=False)
            moved = np.abs(env.datas[0].ctrl - before)
            limit = cfg.servo_speed * env.m.opt.timestep * cfg.n_substeps
            self.assertTrue(np.all(moved <= limit + 1e-9), (moved, limit))
            self.assertTrue(np.all(moved > 0.5 * limit))      # it does move at top speed
            # the policy sees the target that was sent, not the lagging servo
            sent = (env.l0 - env.servo_cmd[0]) / env.stroke
            np.testing.assert_allclose(env.hist[0, -1, 3 + N_TENDON:3 + 2 * N_TENDON],
                                       np.clip(sent, 0, 1))
        finally:
            env.close()

    def test_pull_bias_is_fixed(self):
        cfg = RigConfig(num_envs=3, num_threads=1, episode_steps=20)
        env = SingleLegRigEnv(cfg)
        try:
            for _ in range(2):
                env.reset()
                np.testing.assert_allclose(env.pull_bias, np.tile(env.pull_bias_nominal, (3, 1)))
        finally:
            env.close()

    def test_dr_off_shares_nominal_model(self):
        env = SingleLegRigEnv(RigConfig(num_envs=2, num_threads=1, episode_steps=20))
        try:
            self.assertFalse(env.per_env_models)
            self.assertIs(env.models[0], env.models[1])
            self.assertEqual(env.dr_params(0)["friction"], 1.0)
            np.testing.assert_allclose(env.calib, 0.0)
        finally:
            env.close()

    def test_dr_draws_differ_per_env_and_stay_in_range(self):
        cfg = RigConfig(num_envs=6, num_threads=2, episode_steps=20, dr_enabled=True)
        env = SingleLegRigEnv(cfg)
        try:
            params = [env.dr_params(i) for i in range(6)]
            frictions = [p["friction"] for p in params]
            self.assertGreater(len(set(np.round(frictions, 6))), 1)
            for p in params:
                self.assertTrue(cfg.dr_friction[0] <= p["friction"] <= cfg.dr_friction[1])
                self.assertTrue(cfg.dr_battery[0] <= p["battery"] <= cfg.dr_battery[1])
                self.assertTrue(cfg.dr_servo_kp[0] <= p["servo_kp"] <= cfg.dr_servo_kp[1])
                self.assertTrue(0.0 <= p["delay_ms"] <= 20.0)
            self.assertTrue(np.all(np.abs(env.calib) <= cfg.dr_servo_calib))
            obs, reward, *_ = env.step(torch.zeros(6, env.num_actions))
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            self.assertTrue(torch.isfinite(reward).all())
        finally:
            env.close()

    def test_dr_scale_zero_is_nominal(self):
        env = SingleLegRigEnv(RigConfig(num_envs=2, num_threads=1, episode_steps=20,
                                        dr_enabled=True, dr_scale=0.0))
        try:
            p = env.dr_params(1)
            self.assertAlmostEqual(p["friction"], 1.0)
            self.assertAlmostEqual(p["battery"], 1.0)
            self.assertEqual(p["delay_ms"], 0.0)
        finally:
            env.close()

    def test_full_control_delay_holds_previous_command(self):
        # a 10-substep (20 ms) delay: the first step after a new command
        # still chases the old one, so the servo setpoint does not move
        env = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                        reset_noise=0.0, dr_override={"delay": 10}))
        try:
            env.reset()
            before = env.datas[0].ctrl.copy()
            env.step(torch.ones(1, env.num_actions), auto_reset=False)
            np.testing.assert_allclose(env.datas[0].ctrl, before, atol=1e-12)
            env.step(torch.ones(1, env.num_actions), auto_reset=False)
            self.assertGreater(np.abs(env.datas[0].ctrl - before).max(), 0.0)
        finally:
            env.close()

    def test_voltage_sag_weakens_and_slows_servos(self):
        env = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                        reset_noise=0.0,
                                        dr_override={"battery": 0.5, "sag": 0.0}))
        try:
            env.reset()
            before = env.datas[0].ctrl.copy()
            env.step(torch.ones(1, env.num_actions), auto_reset=False)
            np.testing.assert_allclose(env.models[0].actuator_forcerange[:, 0], -0.5 * env.fmax)
            moved = np.abs(env.datas[0].ctrl - before).max()
            full = env.cfg["servo_speed"] * env.m.opt.timestep * env.n_substeps
            self.assertLessEqual(moved, 0.5 * full + 1e-9)
        finally:
            env.close()

    def test_full_mode_matches_legacy_size(self):
        cfg = RigConfig(num_envs=1, num_threads=1, episode_steps=20, reset_noise=0.0,
                        obs_mode="full")
        env = SingleLegRigEnv(cfg)
        try:
            obs = env.reset()
            self.assertEqual(obs["policy"].shape, (1, 115))
            # the policy never sees the randomized values; the critic does
            torch.testing.assert_close(obs["policy"], obs["critic"][:, :115])
            self.assertEqual(obs["critic"].shape, (1, 127))
        finally:
            env.close()

    def test_critic_sees_randomized_values(self):
        env = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                        dr_override={"friction": 0.4, "battery": 0.8}))
        try:
            obs = env.reset()
            tail = obs["critic"][0, 115:].numpy()
            names = list(env.dr_params(0))
            self.assertAlmostEqual(float(tail[names.index("friction")]), 0.4, places=5)
            self.assertAlmostEqual(float(tail[names.index("battery")]), 0.8, places=5)
            off = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                            critic_dr_params=False))
            try:
                self.assertEqual(off.num_critic_obs, 115)
            finally:
                off.close()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
