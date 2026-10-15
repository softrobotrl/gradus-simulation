"""Random tiled terrain for MuJoCo / MJX, built as one height field.

    terrain = generate("mixed", seed=3, difficulty=0.5, grid=4)
    add_terrain(spec, terrain)          # spec: mujoco.MjSpec; adds the hfield asset and its geom

The ground is a grid of TILE x TILE tiles. Each tile is rough, sloped or covered in blocks ("mixed" draws one kind per
tile at random, any other kind is used everywhere). Every tile border sits at height 0, so neighbouring tiles never
form a cliff. The same arguments always give the same terrain.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

TILE = 2.0  # m, tile side
CELL = 0.05  # m, height-grid spacing
KINDS = ("rough", "slope", "obstacles")  # what "mixed" draws from
# Largest feature at difficulty 1. ASSUMED for a Go2-sized robot (base ~0.27 m above the ground when standing).
ROUGH_AMP = 0.08  # m, bump height (+/-)
SLOPE_DEG = 20.0
SLOPE_TOP = 0.25  # m, half-width of a pyramid's flat top
BLOCK_HEIGHT = 0.12  # m
BASE_DEPTH = 0.1  # m, solid slab under the lowest point so nothing falls through the terrain


@dataclass
class Terrain:
    heights: np.ndarray  # (rows, cols) in m; heights[r, c] sits at x = -size/2 + c*CELL, y = -size/2 + r*CELL
    size: float  # m, side of the square terrain, centred on the world origin
    spawn_xy: tuple[float, float]  # centre of the spawn tile

    def max_height(self, x: float, y: float, radius: float = 0.4) -> float:
        """Highest terrain point within a square of half-width radius around (x, y), which must lie on the terrain."""
        lo = np.floor((np.array([x, y]) - radius + self.size / 2) / CELL).astype(int).clip(0)
        hi = np.ceil((np.array([x, y]) + radius + self.size / 2) / CELL).astype(int) + 1
        return float(self.heights[lo[1]:hi[1], lo[0]:hi[0]].max())


def _tile(kind: str, d: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Heights (n x n) of one tile at difficulty d in [0, 1]. The border rows and columns are exactly 0."""
    u = np.linspace(-1, 1, n)
    x, y = np.meshgrid(u, u)
    border = (1 - np.maximum(abs(x), abs(y))) * TILE / 2  # m from the nearest tile edge
    if kind == "rough":
        h = gaussian_filter(rng.standard_normal((n, n)), sigma=0.1 / CELL)  # sigma 10 cm: bumps ~0.4 m apart
        return h * d * ROUGH_AMP / np.abs(h).max() * np.clip(border / 0.1, 0, 1)  # fade to 0 over the outer 10 cm
    if kind == "slope":
        h = math.tan(math.radians(d * SLOPE_DEG)) * np.minimum(border, TILE / 2 - SLOPE_TOP)
        return h if rng.random() < 0.5 else -h  # hill or pit
    h = np.zeros((n, n))
    if kind == "obstacles":
        lo, hi = round(0.2 / CELL), round(0.5 / CELL)  # 20-50 cm blocks
        for _ in range(rng.integers(5, 16)):
            rows, cols = rng.integers(lo, hi + 1, size=2)
            r, c = rng.integers(1, n - np.array([rows, cols]))  # keeps the border row/column at 0
            h[r:r + rows, c:c + cols] = d * BLOCK_HEIGHT
    return h


def generate(kind: str = "mixed", seed: int | None = None, difficulty: float = 0.5, grid: int = 4) -> Terrain:
    if kind not in ("mixed", "flat", *KINDS):
        raise ValueError(f"unknown terrain kind {kind!r}")
    rng = np.random.default_rng(seed)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    n = round(TILE / CELL)  # cells per tile; neighbouring tiles share their border row/column
    heights = np.zeros((grid * n + 1, grid * n + 1))
    spawn = tuple(int(i) for i in rng.integers(grid, size=2))  # (row, col) of the spawn tile
    for r in range(grid):
        for c in range(grid):
            k = str(rng.choice(KINDS)) if kind == "mixed" else kind
            if kind == "mixed" and (r, c) == spawn:
                k = "flat"  # start the robot on level ground
            heights[r * n:(r + 1) * n + 1, c * n:(c + 1) * n + 1] = _tile(k, rng.uniform(0, difficulty), n + 1, rng)
    size = grid * TILE
    return Terrain(heights, size, ((spawn[1] + 0.5) * TILE - size / 2, (spawn[0] + 0.5) * TILE - size / 2))
