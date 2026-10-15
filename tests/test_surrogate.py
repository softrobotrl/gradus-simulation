import unittest
from unittest.mock import patch

import numpy as np
import torch

from surrogate import Surrogate, SurrogateConfig


class SurrogateTests(unittest.TestCase):
    def test_training_keeps_trajectories_in_separate_partitions(self):
        # Each trajectory has a unique state value, making leakage observable.
        s = np.broadcast_to(np.arange(6)[:, None, None], (6, 4, 1)).copy()
        a = np.zeros_like(s)
        model = Surrogate(SurrogateConfig(state_dim=1, action_dim=1, epochs=1))
        split = model._split_by_sequence
        partitions = []

        def capture(*args):
            result = split(*args)
            partitions.append(result)
            return result

        with patch.object(model, "_split_by_sequence", side_effect=capture):
            model.train(s, a, s + 0.1, verbose=False)
        train, val = partitions[0]
        self.assertTrue(set(train[0].ravel()).isdisjoint(val[0].ravel()))
        self.assertEqual(len(train[0]) + len(val[0]), 24)

    def test_seed_controls_initial_weights(self):
        config = SurrogateConfig(state_dim=2, action_dim=1, seed=42)
        first = Surrogate(config)
        second = Surrogate(config)
        for x, y in zip(first.model.parameters(), second.model.parameters()):
            self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
