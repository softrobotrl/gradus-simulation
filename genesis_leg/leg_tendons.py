"""Forces from tension-only tendon servos, for routes of different lengths.

The same law as the MuJoCo <position> actuators in the merged leg.xml. Genesis doesn't import spatial tendons, so
leg.py applies these forces to the links itself.
"""

import torch


def pad_routes(routes):
    """Stack per-tendon lists of nodes into (tendons, max_nodes, ...) tensors with a (tendons, max_nodes) mask."""
    longest = max(len(route) for route in routes)
    first = torch.as_tensor(routes[0])
    padded = torch.zeros((len(routes), longest, *first.shape[1:]), dtype=first.dtype)
    mask = torch.zeros(len(routes), longest, dtype=torch.bool)
    for i, route in enumerate(routes):
        padded[i, :len(route)] = torch.as_tensor(route)
        mask[i, :len(route)] = True
    return padded, mask


def tendon_forces(points, velocities, mask, ctrl, kp, kv, fmax):
    """Return node forces, tension, length and length rate for (..., tendons, nodes, 3) routed points.

    MuJoCo's actuator force is kp * (ctrl - L) - kv * Ldot, limited to [-fmax, 0]. Here tension is its negative.
    Tension is uniform along each tendon. A slack tendon exerts no force.
    """
    live = mask[..., 1:] & mask[..., :-1]
    segments = points[..., 1:, :] - points[..., :-1, :]
    lengths = torch.linalg.vector_norm(segments, dim=-1).clamp_min(1e-9)
    directions = segments / lengths[..., None]
    rates = ((velocities[..., 1:, :] - velocities[..., :-1, :]) * directions).sum(-1)
    length = (lengths * live).sum(-1)
    length_rate = (rates * live).sum(-1)
    tension = torch.minimum((kp * (length - ctrl) + kv * length_rate).clamp_min(0), fmax)
    segment_force = tension[..., None, None] * directions * live[..., None]
    forces = torch.zeros_like(points)
    forces[..., :-1, :] += segment_force
    forces[..., 1:, :] -= segment_force
    return forces, tension, length, length_rate
