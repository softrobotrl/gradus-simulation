"""Geometry analysis of STL parts: unit detection, canonical framing, tendon-hole detection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


def load_stl(path: Path, units: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if units == "auto":
        # No real part in this pipeline is larger than 2 m; anything bigger is in millimetres.
        units = "mm" if mesh.extents.max() > 2.0 else "m"
    if units == "mm":
        mesh.apply_scale(1e-3)
    return mesh


@dataclass
class DiskFeatures:
    mesh: trimesh.Trimesh  # centred on the backbone axis, mid-plane at z=0, thin axis = z
    thickness: float
    outer_radius: float
    holes: np.ndarray  # (n, 2) tendon-hole centres in the disk frame, sorted by angle
    hole_radius: float


def canonical_disk(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate so the thinnest axis is z, then centre on the outer rim with the mid-plane at z=0."""
    mesh = mesh.copy()
    thin = int(np.argmin(mesh.extents))
    if thin != 2:
        perm = np.eye(4)
        perm[:3, :3] = np.roll(np.eye(3), 2 - thin, axis=0)
        mesh.apply_transform(perm)
    mid_z = mesh.bounds[:, 2].mean()
    loops = _section_loops(mesh, mid_z)
    rim_center, rim_r = max(loops, key=lambda l: l[1])
    # The backbone bore (the loop nearest the rim centre) defines the axis; the rim may have a dent.
    bores = [(c, r) for c, r in loops if r < 0.6 * rim_r and np.linalg.norm(c - rim_center) < 0.1 * rim_r]
    axis = min(bores, key=lambda l: np.linalg.norm(l[0] - rim_center))[0] if bores else rim_center
    mesh.apply_translation([-axis[0], -axis[1], -mid_z])
    return mesh


def _section_loops(mesh: trimesh.Trimesh, z: float) -> list[tuple[np.ndarray, float]]:
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        raise ValueError("disk mesh has no cross-section at its mid-plane")
    loops = []
    for pts in section.discrete:
        # Area centroid and equivalent radius (shoelace), robust to uneven vertex spacing.
        x, y = pts[:, 0], pts[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2
        center = np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * area)
        loops.append((center, float(np.sqrt(abs(area) / np.pi))))
    return loops


def analyse_disk(mesh: trimesh.Trimesh) -> DiskFeatures:
    """Tendon holes = small closed loops off the backbone axis, sharing one radial distance."""
    mesh = canonical_disk(mesh)
    loops = _section_loops(mesh, 0.0)
    outer_r = max(r for _, r in loops)
    candidates = [
        (c, r) for c, r in loops if r < 0.15 * outer_r and np.linalg.norm(c) > 0.3 * outer_r
    ]
    if not candidates:
        raise ValueError("no tendon holes found; set chain.tendon_holes explicitly in the config")
    # Keep the largest group sharing a radial distance (guards against screw holes etc.).
    dists = np.array([np.linalg.norm(c) for c, _ in candidates])
    groups = [np.abs(dists - d) < 0.05 * d for d in dists]
    best = max(groups, key=lambda g: g.sum())
    holes = np.array([c for (c, _), keep in zip(candidates, best) if keep])
    holes = holes[np.argsort(np.arctan2(holes[:, 1], holes[:, 0]))]
    hole_r = float(np.mean([r for (_, r), keep in zip(candidates, best) if keep]))
    return DiskFeatures(mesh, float(mesh.extents[2]), outer_r, holes, hole_r)
