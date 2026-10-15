"""Checks for the random terrain and the Go2 scene built on it.

Usage: python -m pipeline.test_terrain      (first run downloads Go2, ~30 MB)

Runs every check_* function in order and stops at the first failure.
"""

import math

import numpy as np

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


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("check_"):
            check()
            print(f"PASS  {name}")
