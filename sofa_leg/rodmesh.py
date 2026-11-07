"""Hexahedral mesh of a round rod, for SOFA's HexahedronFEMForceField. numpy only.

The cross-section is an n x n grid of nodes on the square [-1, 1]^2, mapped onto the disk with the elliptical grid
mapping (x, y) -> (x sqrt(1 - y^2/2), y sqrt(1 - x^2/2)). Its outline is a polygon, not a circle, so a coarse mesh has
less area and a smaller second moment of area I than the real rod. Two corrections make the coarse mesh bend and
weigh like the real rod:

* the cross-section is scaled so the mesh's I equals the circle's pi r^4 / 4 (bending stiffness EI is then exact for
  the given E), and
* `density_scale` is returned so that density x mesh area equals the real rod's mass per metre.

Torsion and axial stiffness keep a small error from the polygon shape (reported by `section_properties`).

Layers along the rod are placed at caller-given breakpoints (so a disk's faces and centre can be exact node layers),
and each interval between breakpoints is split evenly so no element is longer than `max_length`.
"""

import math

import numpy as np


def _square_to_disk(u, v):
    return u * np.sqrt(1 - v ** 2 / 2), v * np.sqrt(1 - u ** 2 / 2)


def _quads(n):
    """Quads of an n x n node grid, counter-clockwise, as node indices i + n*j."""
    quads = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = i + n * j
            quads.append((a, a + 1, a + 1 + n, a + n))
    return np.array(quads)


def _polygon_moments(xy, quads):
    """Area and second moments (Ixx = integral of y^2, Iyy = integral of x^2) of a quad mesh, exactly."""
    area = ixx = iyy = 0.0
    for quad in quads:
        p = xy[list(quad)]
        for k in range(4):  # shoelace-style integrals over the polygon outline
            x0, y0 = p[k]
            x1, y1 = p[(k + 1) % 4]
            c = x0 * y1 - x1 * y0
            area += c / 2
            ixx += c * (y0 ** 2 + y0 * y1 + y1 ** 2) / 12
            iyy += c * (x0 ** 2 + x0 * x1 + x1 ** 2) / 12
    return area, ixx, iyy


def cross_section(radius, n_side):
    """Cross-section node coordinates (n_side^2, 2), scaled so the mesh's I matches the circle's.

    Returns (xy, quads, density_scale).
    """
    if n_side < 2:
        raise ValueError("n_side must be at least 2")
    s = np.linspace(-1.0, 1.0, n_side)
    u, v = np.meshgrid(s, s, indexing="xy")
    x, y = _square_to_disk(u.ravel(), v.ravel())
    xy = np.stack([x, y], axis=1)
    quads = _quads(n_side)
    _, ixx, iyy = _polygon_moments(xy, quads)
    target = math.pi * radius ** 4 / 4
    xy *= (target / ((ixx + iyy) / 2)) ** 0.25
    area, _, _ = _polygon_moments(xy, quads)
    return xy, quads, math.pi * radius ** 2 / area


def section_properties(radius, n_side):
    """How the meshed cross-section compares with the real circle (ratios, 1.0 = exact)."""
    xy, quads, density_scale = cross_section(radius, n_side)
    area, ixx, iyy = _polygon_moments(xy, quads)
    circle_i = math.pi * radius ** 4 / 4
    return {"area": area / (math.pi * radius ** 2), "I": (ixx + iyy) / 2 / circle_i, "density_scale": density_scale}


def axial_layers(breakpoints, max_length):
    """Sorted node-layer positions along the axis: every breakpoint, plus even splits so no gap exceeds max_length."""
    points = np.unique(np.round(np.asarray(breakpoints, dtype=float), 9))
    layers = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        pieces = max(1, math.ceil((b - a) / max_length - 1e-9))
        layers.extend(a + (b - a) * np.arange(1, pieces + 1) / pieces)
    return np.array(layers)


def rod_mesh(radius, layers, n_side, origin=(0, 0, 0), axis=(0, 0, 1)):
    """Nodes (N, 3) and hexahedra (M, 8) of a straight rod.

    `layers` are positions along `axis` measured from `origin`. Node index = c + n_side^2 * layer, where c is the
    cross-section node. Hexahedra use SOFA's order: the bottom quad counter-clockwise, then the top quad.
    Returns (nodes, hexahedra, density_scale).
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    xy, quads, density_scale = cross_section(radius, n_side)
    per_layer = len(xy)
    nodes = np.concatenate([np.asarray(origin, float) + xy[:, :1] * e1 + xy[:, 1:] * e2 + z * axis for z in layers])
    hexes = []
    for k in range(len(layers) - 1):
        lo, hi = k * per_layer, (k + 1) * per_layer
        for q in quads:
            hexes.append([lo + q[0], lo + q[1], lo + q[2], lo + q[3], hi + q[0], hi + q[1], hi + q[2], hi + q[3]])
    return nodes, np.array(hexes, dtype=int), density_scale
