"""Checks for the random terrain and the Go2 scene built on it.

Usage: python -m pipeline.test_terrain      (first run downloads Go2, ~30 MB)

Runs every check_* function in order and stops at the first failure.
"""

import math

import mujoco
import numpy as np

from pipeline.run_go2 import build_model
from pipeline.terrain import BLOCK_HEIGHT, CELL, ROUGH_AMP, SLOPE_DEG, SLOPE_TOP, TILE, generate


def check_determinism():
    a = generate("mixed", seed=1, difficulty=1.0).heights
    assert np.array_equal(a, generate("mixed", seed=1, difficulty=1.0).heights), "same seed, different terrain"
    assert not np.array_equal(a, generate("mixed", seed=2, difficulty=1.0).heights), "seed has no effect"


def check_borders_and_limits():
    n = round(TILE / CELL)
    limits = {"rough": ROUGH_AMP, "slope": math.tan(math.radians(SLOPE_DEG)) * (TILE / 2 - SLOPE_TOP),
              "obstacles": BLOCK_HEIGHT, "flat": 0.0}
    for kind, limit in limits.items():
        for seed in range(5):
            h = generate(kind, seed=seed, difficulty=1.0).heights
            assert np.abs(h[::n]).max() == 0 and np.abs(h[:, ::n]).max() == 0, f"{kind}: cliff at a tile border"
            assert np.abs(h).max() <= limit + 1e-9, f"{kind}: {np.abs(h).max():.3f} m exceeds {limit:.3f} m"
            assert kind == "flat" or np.abs(h).max() > 0, f"{kind}: terrain is flat"
            assert not generate(kind, seed=seed, difficulty=0.0).heights.any(), f"{kind}: difficulty 0 is not flat"


def check_spawn_tile_is_flat():
    for seed in range(20):
        t = generate("mixed", seed=seed, difficulty=1.0)
        assert t.max_height(*t.spawn_xy) == 0, f"seed {seed}: spawn point {t.spawn_xy} is not on the flat tile"


def check_compiles():
    for kind in ("mixed", "rough", "slope", "obstacles", "flat"):
        build_model(generate(kind, seed=0))


def check_heights_match_mujoco():
    t = generate("mixed", seed=5, difficulty=1.0)
    model = build_model(t)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    only_terrain = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # terrain is geom group 0; Go2 uses groups 2 and 3
    n = t.heights.shape[0]
    for r, c in np.random.default_rng(0).integers(1, n - 1, size=(200, 2)):
        # 1 um off the grid vertex: mj_ray can miss the surface when a ray passes exactly through a vertex
        x, y = -t.size / 2 + c * CELL + 1e-6, -t.size / 2 + r * CELL + 1e-6
        dist = mujoco.mj_ray(model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]), only_terrain, 1, -1,
                             np.zeros(1, np.int32))
        assert abs(5.0 - dist - t.heights[r, c]) < 1e-4, f"MuJoCo height {5.0 - dist:.4f} != {t.heights[r, c]:.4f}"


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("check_"):
            check()
            print(f"PASS  {name}")
