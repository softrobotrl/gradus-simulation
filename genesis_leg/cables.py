"""Spring and damper forces along a cable routed through guides."""

import torch


def rotate(quat, vector):
    """Rotate vectors by unit wxyz quaternions, with broadcasting."""
    xyz = quat[..., 1:].expand_as(vector)
    uv = torch.linalg.cross(xyz, vector)
    return vector + 2 * (quat[..., :1] * uv + torch.linalg.cross(xyz, uv))


def cable_forces(points, velocities, rest_length, stiffness, damping):
    """Return node forces and tension for (..., nodes, 3) routed cable points.

    Tension is uniform along each cable. Slack cables exert no force.
    """
    segments = points[..., 1:, :] - points[..., :-1, :]
    lengths = torch.linalg.vector_norm(segments, dim=-1).clamp_min(1e-9)
    directions = segments / lengths[..., None]
    relative_velocity = velocities[..., 1:, :] - velocities[..., :-1, :]
    length_rate = (relative_velocity * directions).sum((-1, -2))
    extension = lengths.sum(-1) - rest_length
    tension = torch.where(
        extension > 0,
        (stiffness * extension + damping * length_rate).clamp_min(0),
        0,
    )
    segment_force = tension[..., None, None] * directions
    forces = torch.zeros_like(points)
    forces[..., :-1, :] += segment_force
    forces[..., 1:, :] -= segment_force
    return forces, tension
