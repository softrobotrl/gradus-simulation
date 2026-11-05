"""Terrain generator and Genesis scene checks. Builds a real scene, so expect about a minute on first run."""

import math
import unittest

import genesis as gs
import numpy as np

from pipeline.run_go2_genesis import add_go2, add_terrain, hold_home_pose, new_scene
from pipeline.terrain import BLOCK_HEIGHT, CELL, ROUGH_AMP, SLOPE_DEG, SLOPE_TOP, TILE, generate


class GeneratorTests(unittest.TestCase):
    def test_seed_sets_the_terrain(self):
        heights = generate("mixed", seed=1, difficulty=1.0).heights
        self.assertTrue(np.array_equal(heights, generate("mixed", seed=1, difficulty=1.0).heights))
        self.assertFalse(np.array_equal(heights, generate("mixed", seed=2, difficulty=1.0).heights))

    def test_tile_borders_and_height_limits(self):
        n = round(TILE / CELL)
        limits = {"rough": ROUGH_AMP, "slope": math.tan(math.radians(SLOPE_DEG)) * (TILE / 2 - SLOPE_TOP),
                  "obstacles": BLOCK_HEIGHT, "flat": 0.0}
        for kind, limit in limits.items():
            for seed in range(5):
                heights = generate(kind, seed=seed, difficulty=1.0).heights
                self.assertEqual(np.abs(heights[::n]).max(), 0, kind)
                self.assertEqual(np.abs(heights[:, ::n]).max(), 0, kind)
                self.assertLessEqual(np.abs(heights).max(), limit + 1e-9, kind)
                self.assertFalse(generate(kind, seed=seed, difficulty=0.0).heights.any(), kind)

    def test_spawn_tile_is_flat(self):
        for kind in ("mixed", "rough", "slope", "obstacles"):
            for seed in range(10):
                terrain = generate(kind, seed=seed, difficulty=1.0)
                self.assertEqual(terrain.max_height(*terrain.spawn_xy), 0, f"{kind} seed {seed}")


class GenesisSceneTests(unittest.TestCase):
    BALL_RADIUS = 0.05

    @classmethod
    def setUpClass(cls):
        if not gs._initialized:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=0)
        # At this seed the 21 cm hill at (-3, 1) and the 17 cm pit at (1, -3) are mirror images across y = x,
        # so a height map with x and y swapped puts each ball at the wrong height.
        cls.terrain = generate("mixed", seed=3, difficulty=1.0)
        scene = new_scene(cls.terrain)
        cls.ground = add_terrain(scene, cls.terrain)
        cls.robot = add_go2(scene, cls.terrain)
        cls.balls = {xy: scene.add_entity(gs.morphs.Sphere(radius=cls.BALL_RADIUS, pos=(*xy, 0.5)))
                     for xy in ((-3.0, 1.0), (1.0, -3.0))}
        scene.build()
        hold_home_pose(cls.robot)
        for _ in range(100):
            scene.step()
        cls.solver_error = bool(scene.rigid_solver.get_error_envs_mask().any())

    def test_balls_rest_at_the_generated_heights(self):
        terrain = self.terrain
        for (x, y), ball in self.balls.items():
            row, col = round((y + terrain.size / 2) / CELL), round((x + terrain.size / 2) / CELL)
            rest = float(ball.get_pos().ravel()[2]) - self.BALL_RADIUS
            self.assertAlmostEqual(rest, terrain.heights[row, col], delta=0.005, msg=f"ball at ({x}, {y})")
        self.assertGreater(terrain.heights[round((1 + 4) / CELL), round((-3 + 4) / CELL)], 0.15)

    def test_height_query_matches_the_generated_grid(self):
        axis = -self.terrain.size / 2 + np.arange(self.terrain.heights.shape[0]) * CELL
        x, y = np.meshgrid(axis, axis)
        heights = self.ground.get_terrain_height(np.stack((x.ravel(), y.ravel()), -1)).cpu().numpy()
        np.testing.assert_allclose(heights.reshape(self.terrain.heights.shape), self.terrain.heights, atol=1e-4)

    def test_go2_stands_on_the_spawn_tile(self):
        base = self.robot.get_pos().cpu().numpy().ravel()
        self.assertFalse(self.solver_error)
        self.assertLess(np.linalg.norm(base[:2] - self.terrain.spawn_xy), 0.15, "base left the spawn point")
        self.assertGreater(base[2], 0.15, "base fell")
        links = set(self.robot.get_contacts(with_entity=self.ground)["link_a"].tolist())
        links |= set(self.robot.get_contacts(with_entity=self.ground)["link_b"].tolist())
        calves = {self.robot.get_link(f"{leg}_calf").idx for leg in ("FL", "FR", "RL", "RR")}
        self.assertEqual(links & calves, calves, "not all four feet are on the terrain")


if __name__ == "__main__":
    unittest.main()
