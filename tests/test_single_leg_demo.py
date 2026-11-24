"""Small checks for the single demo leg rig; the env test is skipped until built."""

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from onshape_to_mjcf.merge import _dir_of, _slug
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

    def test_clipped_model_bounds_normalized_inputs(self):
        from tensordict import TensorDict
        from single_leg_demo.models import OBS_CLIP, ClippedMLPModel
        obs = TensorDict({"policy": torch.zeros(4, 3)}, batch_size=[4])
        model = ClippedMLPModel(obs, {"actor": ["policy"]}, "actor", 2,
                                hidden_dims=[8], obs_normalization=True)
        model.obs_normalizer.update(torch.randn(200, 3) * 0.004)   # tiny sim std
        latent = model.get_latent(TensorDict({"policy": torch.full((1, 3), 0.1)},
                                             batch_size=[1]))
        self.assertTrue(torch.all(latent.abs() <= OBS_CLIP))

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
            self.assertEqual(obs["critic"].shape, (2, 115 + 1 + 13))   # state + scored speed + DR
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

    def test_rail_height_randomized_within_range(self):
        cfg = RigConfig(num_envs=6, num_threads=2, episode_steps=20, dr_enabled=True)
        env = SingleLegRigEnv(cfg)
        try:
            heights = [env.dr_params(i)["rail_height_mm"] for i in range(6)]
            lo, hi = (1000 * v for v in cfg.dr_rail_height)
            self.assertTrue(all(lo - 1e-9 <= h <= hi + 1e-9 for h in heights), heights)
            self.assertGreater(len(set(np.round(heights, 6))), 1)
            # a lower rail moves the hip (carriage) down by exactly that much
            low = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                            dr_override={"rail_height": -0.005}))
            try:
                z = low.datas[0].xpos[low.slider_body, 2]
                nominal = low.nom["body_pos"][low.slider_body, 2]
                self.assertAlmostEqual(float(z), float(nominal) - 0.005, places=9)
            finally:
                low.close()
        finally:
            env.close()

    def test_delay_longer_than_one_step(self):
        # 30 ms = 15 substeps: nothing in step 1; in step 2 the first command
        # arrives after 5 substeps, so the servo moves for only 5 of 10
        env = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                        reset_noise=0.0, dr_override={"delay": 15}))
        try:
            env.reset()
            before = env.datas[0].ctrl.copy()
            env.step(torch.ones(1, env.num_actions), auto_reset=False)
            np.testing.assert_allclose(env.datas[0].ctrl, before, atol=1e-12)
            env.step(torch.ones(1, env.num_actions), auto_reset=False)
            moved = np.abs(env.datas[0].ctrl - before).max()
            per_substep = env.cfg["servo_speed"] * env.m.opt.timestep
            self.assertAlmostEqual(moved, 5 * per_substep, places=9)
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
            self.assertEqual(obs["critic"].shape, (1, 129))
        finally:
            env.close()

    def test_critic_sees_randomized_values(self):
        env = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                        dr_override={"friction": 0.4, "battery": 0.8}))
        try:
            obs = env.reset()
            tail = obs["critic"][0, 116:].numpy()   # after the 115 state values and the scored speed
            names = list(env.dr_params(0))
            self.assertAlmostEqual(float(tail[names.index("friction")]), 0.4, places=5)
            self.assertAlmostEqual(float(tail[names.index("battery")]), 0.8, places=5)
            off = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=20,
                                            critic_dr_params=False))
            try:
                self.assertEqual(off.num_critic_obs, 116)
            finally:
                off.close()
        finally:
            env.close()

    def test_foot_sensors_match_contacts(self):
        # The reward reads the foot sensors by name; rig.xml has other sensors
        # (the rail IMU) first, so a fixed index would read the wrong values.
        env = SingleLegRigEnv(RigConfig(num_envs=1, num_threads=1, episode_steps=200,
                                        reset_noise=0.0))
        try:
            env.reset()
            f6, vel = np.zeros(6), np.zeros(6)
            rng = np.random.default_rng(0)
            touched = 0
            for _ in range(60):
                env.step(torch.from_numpy(rng.uniform(-1, 1, (1, env.num_actions)).astype(np.float32)))
                d = env.datas[0]
                fn_s, slip_s = env._foot_sensor(d)
                fn_c, slip_c = env._foot_contact(d, f6, vel)
                self.assertAlmostEqual(fn_s, fn_c, delta=1e-6 + 1e-6 * abs(fn_c))
                if fn_c > 1e-6:
                    touched += 1
                    # sensor: slip at the sphere's lowest point; reference: at MuJoCo's
                    # contact point, slightly inside the penetration (a few % apart)
                    self.assertAlmostEqual(slip_s, slip_c, delta=1e-3 + 0.05 * slip_c)
            self.assertGreater(touched, 10)
        finally:
            env.close()


class RewardModuleTests(unittest.TestCase):
    def test_presets_and_overrides(self):
        from single_leg_demo import rewards
        cfg = rewards.apply_preset(RigConfig(), "v4")
        self.assertEqual((cfg.vel_avg_tau, cfg.vel_sigma, cfg.reward_preset), (1.0, 0.03, "v4"))
        rewards.apply_overrides(cfg, ["accel_weight=1.5", "bound_costs=false",
                                      "obs_history=4", "dr_friction=(0.3, 1.0)"])
        self.assertEqual((cfg.accel_weight, cfg.bound_costs, cfg.obs_history, cfg.dr_friction),
                         (1.5, False, 4, (0.3, 1.0)))
        with self.assertRaises(SystemExit):
            rewards.apply_overrides(cfg, ["accel_wieght=1"])     # typo is an error, not a no-op
        for preset in rewards.PRESETS.values():                   # presets only touch real fields
            self.assertTrue(set(preset) <= {f for f in vars(RigConfig())})
        self.assertTrue(rewards.REWARD_FIELDS <= set(vars(RigConfig())))


@unittest.skipUnless(RIG_XML.exists(), "build the rig first: python -m single_leg_demo.build_rig")
class RewardBehaviourTests(unittest.TestCase):
    def test_logged_terms_add_up_to_the_reward(self):
        from single_leg_demo import rewards
        cfg = rewards.apply_preset(RigConfig(num_envs=1, num_threads=1, episode_steps=100,
                                             bound_costs=False), "v4")
        env = SingleLegRigEnv(cfg)
        try:
            env.reset()
            rng = np.random.default_rng(0)
            for _ in range(40):
                act = torch.from_numpy(rng.uniform(-1, 1, (1, env.num_actions)).astype(np.float32))
                _, r, done, extras = env.step(act)
                if bool(done[0]):
                    break
                parts = sum(v for k, v in extras["log"].items() if k.startswith("/r_"))
                self.assertAlmostEqual(parts, float(r[0]), places=4)
        finally:
            env.close()

    def test_preference_v4_prefers_walking_v1_standing(self):
        # The core check behind reward v4 (REWARD.md): scored on the same physics,
        # v4 must rank the scripted walk above standing still; v1 did the opposite.
        from single_leg_demo.reward_check import preference as pref
        for preset, want in (("v4", "PASS"), ("v5", "PASS"), ("v1", "FAIL")):
            cfg = pref.make_config(preset, [], rail=3.0, cmd=0.07, period=4.0, steps=600)
            label, why = pref.verdict(pref.score(cfg))
            self.assertEqual(label, want, f"{preset}: {why}")

    def test_idle_term(self):
        # v5's idle cost: 0 when the ~0.5 s average speed keeps up with the
        # command, -1 when standing or going the wrong way, 0 with no command.
        from dataclasses import fields
        from single_leg_demo import rewards
        s = rewards.StepState(**{f.name: 0 for f in fields(rewards.StepState)})
        c = {"cmd_min": 0.05}
        for cmd, v, want in ((0.07, 0.07, 0.0), (0.07, 0.10, 0.0), (0.07, 0.0, -1.0),
                             (0.07, -0.05, -1.0), (-0.07, -0.035, -0.5), (0.0, 0.0, 0.0)):
            s.cmd, s.v_idle = cmd, v
            self.assertAlmostEqual(rewards.idle(s, c), want, places=6)


if __name__ == "__main__":
    unittest.main()
