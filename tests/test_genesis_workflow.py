"""Checkpoint recovery and evaluation across asynchronous episode endings."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from genesis_leg.evaluate import evaluate
from genesis_leg.runner import LegPPO, LegRunner


class TimeoutTests(unittest.TestCase):
    def test_timeout_uses_final_state_and_failure_does_not_bootstrap(self):
        algorithm = LegPPO.__new__(LegPPO)
        algorithm.gamma = 0.9
        algorithm.device = "cpu"
        algorithm.rnd = None
        algorithm.actor = Mock()
        algorithm.critic = Mock(side_effect=lambda obs: obs["policy"][:, :1])
        algorithm.transition = RolloutStorage.Transition()
        algorithm.transition.values = torch.tensor([[2.], [3.], [4.]])
        stored = {}

        def record(transition):
            stored["rewards"] = transition.rewards.clone()
            stored["values"] = transition.values.clone()

        algorithm.storage = SimpleNamespace(add_transition=record)
        reset_obs = TensorDict({"policy": torch.full((3, 1), 999.)}, batch_size=[3])
        extras = {
            "time_outs": torch.tensor([1., 0., 0.]),
            "final_observation": torch.tensor([[10.], [20.], [30.]]),
        }
        rewards = torch.tensor([1., 2., 3.])
        algorithm.process_env_step(reset_obs, rewards, torch.tensor([True, True, False]), extras)
        torch.testing.assert_close(stored["rewards"], torch.tensor([10., 2., 3.]))
        torch.testing.assert_close(stored["values"], torch.tensor([[2.], [3.], [4.]]))
        torch.testing.assert_close(rewards, torch.tensor([1., 2., 3.]))
        self.assertIn("time_outs", extras)


class CheckpointTests(unittest.TestCase):
    def make_runner(self, iteration):
        runner = LegRunner.__new__(LegRunner)
        runner.alg = SimpleNamespace(save=lambda: {"weights": torch.tensor([iteration])})
        runner.current_learning_iteration = iteration
        runner.logger = Mock()
        return runner

    def test_periodic_save_refreshes_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_50.pt"
            self.make_runner(50).save(path)
            for checkpoint in (path, path.with_name("latest.pt")):
                self.assertEqual(torch.load(checkpoint, weights_only=True)["iter"], 50)

    def test_interrupted_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            self.make_runner(50).save(path)
            previous = path.read_bytes()

            def interrupted_save(checkpoint, destination):
                Path(destination).write_bytes(b"partial write")
                raise OSError("disk full")

            with patch("genesis_leg.runner.torch.save", side_effect=interrupted_save):
                with self.assertRaises(OSError):
                    self.make_runner(100).save(path)
            self.assertEqual(path.read_bytes(), previous)


class EvaluationTests(unittest.TestCase):
    def test_finished_env_does_not_contribute_later_motion_or_tension(self):
        class Env:
            num_envs = 2
            max_episode_length = 3
            dt = 1.0
            device = "cpu"
            board_qs = [0, 1, 2]

            def __init__(self):
                self.robot = SimpleNamespace(get_qpos=lambda: torch.zeros(2, 3))

            def reset(self):
                self.steps = 0
                return TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])

            def step(self, actions):
                self.steps += 1
                # Env zero ends first, then starts an unrelated high-tension episode.
                x = 1.0 if self.steps == 1 else 100.0
                final_obs = torch.tensor([[x, 0., 0.], [float(self.steps), 0., 0.]])
                extras = {
                    "final_observation": final_obs,
                    "time_outs": torch.ones(2),
                    "peak_tension": torch.tensor([2. if self.steps == 1 else 999., 3.]),
                }
                done = torch.tensor([self.steps == 1, self.steps == 3])
                return self.reset_observations(), torch.ones(2), done, extras

            def reset_observations(self):
                # Auto-reset observations differ from final observations.
                return TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])

        result = evaluate(Env(), lambda obs, time: torch.zeros(2, 3))
        self.assertEqual(result["peak_tension_N"], 3.)
        self.assertEqual(result["board_speed_mm_s"], 1000.)
        self.assertEqual(result["episode_seconds_mean"], 2.)
        self.assertEqual(result["return_mean"], 2.)
        self.assertEqual(result["failure_rate"], 0.)


if __name__ == "__main__":
    unittest.main()
