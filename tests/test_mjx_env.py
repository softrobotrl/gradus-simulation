"""MJX env vs the CPU reference (needs jax, mujoco.mjx and brax; skipped otherwise).

    python -m unittest tests.test_mjx_env
"""

import dataclasses
import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from single_leg_demo import mjx_env, mjx_rewards
    HAVE_JAX = True
except ImportError:          # the CPU-only environment
    HAVE_JAX = False

import torch

from single_leg_demo import rewards
from single_leg_demo.rig_env import N_TENDON, RigConfig, SingleLegRigEnv


def random_state(rng, n_act):
    t = lambda *s: rng.normal(size=s) if s else float(rng.normal())
    return dict(
        t=int(rng.integers(0, 6)), cmd=float(rng.choice([0.0, 0.07, -0.05, 1e-7])) + 0.01 * t(),
        vx=t() * 0.1, v_track=t() * 0.1, v_idle=t() * 0.1, x=float(rng.uniform(-0.45, 0.45)),
        foot_z=float(rng.uniform(0.0, 0.1)), foot_y=t() * 0.1, bend=np.abs(rng.normal(size=13)) * 0.5,
        pull=rng.uniform(0, 1, N_TENDON), force=rng.uniform(0, 1.2, N_TENDON),
        tension=rng.uniform(0, 200, N_TENDON), prev_tension=rng.uniform(0, 200, N_TENDON),
        peak_tension=rng.uniform(0, 220, N_TENDON), fmax=np.full(N_TENDON, 200.0),
        action=rng.uniform(-1, 1, n_act), action_raw=rng.normal(size=n_act) * 1.5,
        prev_action=rng.uniform(-1, 1, n_act), prev_action2=rng.uniform(-1, 1, n_act),
        foot_force_peak=float(rng.uniform(0, 150)), slip=float(rng.uniform(0, 0.5)))


@unittest.skipUnless(HAVE_JAX, "jax / brax not installed")
class MjxRewardTests(unittest.TestCase):
    def test_every_term_matches_rewards_py(self):
        jax.config.update("jax_enable_x64", True)
        rng = np.random.default_rng(0)
        for preset in ("v1", "v4", "v5"):
            cfg = dataclasses.asdict(rewards.apply_preset(RigConfig(), preset))
            for _ in range(200):
                d = random_state(rng, 4)
                ref, ref_parts, ref_clip = rewards.total(rewards.StepState(**d), cfg, 0.7)
                got, parts, clip = mjx_rewards.total(
                    mjx_rewards.StepState(**{k: jnp.asarray(v) for k, v in d.items()}), cfg, 0.7)
                for name, v in ref_parts.items():
                    self.assertAlmostEqual(float(parts[name]), v, places=9, msg=f"{preset} {name}")
                self.assertAlmostEqual(float(got), ref, places=9, msg=preset)
                self.assertEqual(bool(clip), bool(ref_clip))


@unittest.skipUnless(HAVE_JAX, "jax / brax not installed")
class MjxEnvTests(unittest.TestCase):
    def _pair(self, **over):
        cfg = RigConfig(num_envs=1, num_threads=1, rail_friction=3.0, cmd_resample=False,
                        cmd_period_jitter=0.0, episode_steps=400, **over)
        rewards.apply_preset(cfg, "v5")
        cpu = SingleLegRigEnv(cfg)
        cpu.reset()
        # a one-entry bank holding exactly the CPU env's episode start
        bank = {}
        d, mi = cpu.datas[0], cpu.models[0]
        for k in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            bank[k] = getattr(d, k)[None].copy()
        for k in mjx_env.MODEL_FIELDS:
            bank[k] = getattr(mi, k)[None].copy()
        for k in mjx_env.DRAW_FIELDS:
            bank[k] = np.array(getattr(cpu, k)[0], dtype=np.float64)[None]
        env = mjx_env.MjxRigEnv(cfg, bank=bank)
        st = env.reset(jax.random.PRNGKey(0))
        # same command as the CPU env
        rig = dict(st.info["rig"])
        for k in ("cmd", "cmd_raw", "cmd_mag", "cmd_sign0", "cmd_period"):
            rig[k] = jnp.asarray(getattr(cpu, k)[0])
        st = st.replace(info={**st.info, "rig": rig})
        return cpu, env, st

    def test_steps_match_cpu_env(self):
        jax.config.update("jax_enable_x64", True)
        cpu, env, st = self._pair()
        step = jax.jit(env.step)
        rng = np.random.default_rng(3)
        worst = {"qpos": 0.0, "obs": 0.0, "acc": 0.0, "critic": 0.0, "reward": 0.0}
        F = env.hw_frame
        acc = slice(F - 6, F - 3)                   # rail IMU accelerometer in each frame
        for t in range(60):
            a = rng.uniform(-1, 1, (1, cpu.num_actions)).astype(np.float32)
            o, r, done, _ = cpu.step(torch.from_numpy(a), auto_reset=False)
            st = step(st, jnp.asarray(a[0], dtype=jnp.float64))
            worst["qpos"] = max(worst["qpos"], float(np.max(np.abs(np.array(st.pipeline_state.qpos) - cpu.datas[0].qpos))))
            diff = np.abs(np.array(st.obs["state"]) - o["policy"][0].numpy()).reshape(-1, F)
            worst["acc"] = max(worst["acc"], float(diff[:, acc].max()))
            diff[:, acc] = 0.0
            worst["obs"] = max(worst["obs"], float(diff.max()))
            worst["critic"] = max(worst["critic"], float(np.max(np.abs(np.array(st.obs["privileged_state"]) - o["critic"][0].numpy()))))
            worst["reward"] = max(worst["reward"], abs(float(st.reward) - float(r[0])))
            if bool(done[0]):
                break
        cpu.close()
        print("worst differences over 60 steps:", worst)
        self.assertLess(worst["qpos"], 1e-4)
        self.assertLess(worst["reward"], 1e-3)
        self.assertLess(worst["critic"], 1e-3)
        self.assertLess(worst["obs"], 2e-3)        # encoder rounding can flip one count
        # the accelerometer reads the solver's acceleration, the most sensitive
        # quantity; ~0.01 g is the sensor noise added in training anyway
        self.assertLess(worst["acc"], 0.02)


if __name__ == "__main__":
    unittest.main()
