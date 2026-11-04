"""Random 2-D shapes for tip tracing.

Shapes are sampled on a unit-ish grid and normalised into a disk of the
requested radius.  The distribution mixes:

* 50 % random Fourier curves (smooth, closed, star-like and lobed),
* 30 % random polygons (sharp corners, 6 vertices, lightly rounded),
* 20 % quadratic Bezier arcs (open strokes, traced forward then back).

The policy never sees the whole shape -- only a short look-ahead window --
so it has to learn to follow arbitrary curves rather than memorise one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi


def _normalise(xy: jnp.ndarray, radius: jnp.ndarray) -> jnp.ndarray:
    """Centre and scale a curve so its furthest point is at ``radius``."""
    xy = xy - xy.mean(axis=0, keepdims=True)
    scale = jnp.max(jnp.linalg.norm(xy, axis=-1))
    return xy * (radius / (scale + 1e-6))


def _rotate(xy: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    ang = jax.random.uniform(key, (), minval=0.0, maxval=TWO_PI)
    c, s = jnp.cos(ang), jnp.sin(ang)
    R = jnp.array([[c, -s], [s, c]])
    return xy @ R.T


def fourier_shape(key: jax.Array, n_points: int, radius: float) -> jnp.ndarray:
    k_a, k_r, k_s = jax.random.split(key, 3)
    harmonics = 3
    t = jnp.linspace(0.0, 1.0, n_points, endpoint=False)
    n = jnp.arange(1, harmonics + 1, dtype=jnp.float32)[:, None]
    c = jnp.cos(TWO_PI * n * t[None, :])
    s = jnp.sin(TWO_PI * n * t[None, :])
    a = jax.random.normal(k_a, (harmonics, 2, 2))
    a = a / (jnp.arange(1, harmonics + 1, dtype=jnp.float32) ** 1.1)[:, None, None]
    x = jnp.sum(a[:, 0, 0, None] * c + a[:, 0, 1, None] * s, axis=0)
    y = jnp.sum(a[:, 1, 0, None] * c + a[:, 1, 1, None] * s, axis=0)
    xy = jnp.stack([x, y], axis=-1)
    xy = _rotate(xy, k_r)
    sc = jax.random.uniform(k_s, (), minval=0.65, maxval=1.0) * radius
    return _normalise(xy, sc)


def polygon_shape(key: jax.Array, n_points: int, radius: float) -> jnp.ndarray:
    k_a, k_r, k_s = jax.random.split(key, 3)
    nv = 6
    ang = jnp.sort(jax.random.uniform(k_a, (nv,), minval=0.0, maxval=TWO_PI))
    rr = jax.random.uniform(k_r, (nv,), minval=0.55, maxval=1.0) * radius
    pts = jnp.stack([rr * jnp.cos(ang), rr * jnp.sin(ang)], axis=-1)
    pts = jnp.concatenate([pts, pts[:1]], axis=0)

    seg = jnp.linalg.norm(jnp.diff(pts, axis=0), axis=-1)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    total = cum[-1]
    u = jnp.linspace(0.0, total, n_points, endpoint=False)
    idx = jnp.clip(jnp.searchsorted(cum, u, side="right") - 1, 0, nv - 1)
    frac = (u - cum[idx]) / (seg[idx] + 1e-9)
    xy = pts[idx] + frac[:, None] * (pts[idx + 1] - pts[idx])

    # round the corners a little
    for _ in range(2):
        xy = 0.25 * jnp.roll(xy, 1, axis=0) + 0.5 * xy + 0.25 * jnp.roll(xy, -1, axis=0)

    sc = jax.random.uniform(k_s, (), minval=0.65, maxval=1.0) * radius
    return _normalise(xy, sc)


def bezier_shape(key: jax.Array, n_points: int, radius: float) -> jnp.ndarray:
    k_p, k_r, k_s = jax.random.split(key, 3)
    direction = jax.random.normal(k_p, (3, 2))
    direction = direction / (jnp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6)
    scales = jnp.array([0.55, 1.0, 0.7]) * radius
    pts = direction * scales[:, None]

    t = jnp.linspace(0.0, 1.0, n_points)
    xy = ((1.0 - t)[:, None] ** 2) * pts[0]
    xy = xy + 2.0 * ((1.0 - t) * t)[:, None] * pts[1]
    xy = xy + (t[:, None] ** 2) * pts[2]
    xy = _rotate(xy, k_r)
    sc = jax.random.uniform(k_s, (), minval=0.65, maxval=1.0) * radius
    return _normalise(xy, sc)


def sample_shape(key: jax.Array, n_points: int, radius: float):
    """Return ``(shape (n_points, 2), closed: bool)`` for one episode."""
    k_fam, k_f, k_p, k_b = jax.random.split(key, 4)
    fam = jax.random.uniform(k_fam, ())
    closed = fam < 0.80
    f = fourier_shape(k_f, n_points, radius)
    p = polygon_shape(k_p, n_points, radius)
    b = bezier_shape(k_b, n_points, radius)
    shape = jnp.where(fam < 0.50, f, jnp.where(fam < 0.80, p, b))
    return shape, closed


def phase_to_u(phase: jnp.ndarray, closed: jnp.ndarray) -> jnp.ndarray:
    """Map a phase in [0,1) to the curve parameter.

    Closed curves wrap; open strokes ping-pong so the tip never teleports
    from the end of the stroke back to the start.
    """
    return jnp.where(closed, phase, 1.0 - jnp.abs(2.0 * phase - 1.0))


def target_at(shape: jnp.ndarray, phase: jnp.ndarray, closed: jnp.ndarray,
              n_points: int) -> jnp.ndarray:
    """Interpolated target point for one shape at one phase."""
    u = phase_to_u(phase, closed) % 1.0
    idx = u * n_points
    i0 = jnp.floor(idx).astype(jnp.int32) % n_points
    i1 = (i0 + 1) % n_points
    frac = idx - jnp.floor(idx)
    return shape[i0] + frac * (shape[i1] - shape[i0])


def target_window(shape: jnp.ndarray, phase: jnp.ndarray, closed: jnp.ndarray,
                  lookahead: int, lookahead_dt: float, period: float,
                  n_points: int) -> jnp.ndarray:
    """``(lookahead+1, 2)`` current and future target points."""
    offs = jnp.arange(lookahead + 1, dtype=jnp.float32) * (lookahead_dt / period)
    phases = (phase + offs) % 1.0
    return jax.vmap(lambda ph: target_at(shape, ph, closed, n_points))(phases)


# --------------------------------------------------------------------------- #
# Held-out shapes for evaluation
# --------------------------------------------------------------------------- #
def circle(n_points: int, radius: float) -> jnp.ndarray:
    t = jnp.linspace(0.0, 1.0, n_points, endpoint=False)
    return jnp.stack([jnp.cos(TWO_PI * t), jnp.sin(TWO_PI * t)], axis=-1) * radius


def square(n_points: int, radius: float) -> jnp.ndarray:
    t = jnp.linspace(0.0, 1.0, n_points, endpoint=False) * 4.0
    x = jnp.where(t < 1.0, t, jnp.where(t < 2.0, 1.0, jnp.where(t < 3.0, 3.0 - t, 0.0)))
    y = jnp.where(t < 1.0, 0.0, jnp.where(t < 2.0, t - 1.0, jnp.where(t < 3.0, 1.0, 4.0 - t)))
    xy = jnp.stack([x, y], axis=-1) - 0.5
    return xy * (2.0 * radius)


def star(n_points: int, radius: float) -> jnp.ndarray:
    t = jnp.linspace(0.0, 1.0, n_points, endpoint=False)
    r = jnp.where((t * 10.0).astype(jnp.int32) % 2 == 0, 1.0, 0.45) * radius
    a = TWO_PI * t - jnp.pi / 2.0
    return jnp.stack([r * jnp.cos(a), r * jnp.sin(a)], axis=-1)


def figure_eight(n_points: int, radius: float) -> jnp.ndarray:
    t = jnp.linspace(0.0, 1.0, n_points, endpoint=False)
    xy = jnp.stack([jnp.sin(TWO_PI * t), jnp.sin(2.0 * TWO_PI * t) * 0.5], axis=-1)
    return _normalise(xy, jnp.asarray(radius))


def eval_shapes(radius: float, n_points: int) -> dict[str, jnp.ndarray]:
    return {
        "circle": circle(n_points, radius),
        "square": square(n_points, radius),
        "star": star(n_points, radius),
        "figure_eight": figure_eight(n_points, radius),
    }
