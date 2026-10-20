"""Soft "fishing-line" tendon on a Go2 leg, simulated as a PBD (XPBD) ribbon.

The tendon is a thin 1D ribbon of particles whose ends are pinned to the robot
links with ``PBDBaseEntity.fix_particles_to_link``. This gives a genuinely soft,
stretch-compliant line that follows the leg; the leg drives the tendon (rigid ->
PBD coupling).

Note
----
Pinning is one-way: the link drives the pinned particles. If you need the tendon
to *pull* the leg (tendon-driven actuation), apply the tendon force yourself in
the controller - see ``tendon_torque`` below - or use a rigid articulated cable.

Usage
-----
    python genesis_samples/go2_tendon.py            # headless
    python genesis_samples/go2_tendon.py --viewer   # open the viewer
"""

import argparse

import numpy as np
import torch
import trimesh

import genesis as gs


def make_ribbon(path, n_nodes=60, length=0.7, width=4e-3):
    """Build a straight strip mesh; it gets warped onto the leg after build()."""
    verts, faces = [], []
    for i in range(n_nodes):
        x = length * i / (n_nodes - 1)
        verts += [[x, -width / 2, 0.0], [x, width / 2, 0.0]]
    for i in range(n_nodes - 1):
        a, b = 2 * i, 2 * i + 2
        faces += [[a, b, a + 1], [b, b + 1, a + 1]]
    trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=False).export(path)


def warp_along(points, width_dir):
    """Resample a polyline and turn it into the ribbon's (2 * n_nodes, 3) vertices."""
    n = points.shape[0]
    return np.stack([points, points + width_dir], axis=1).reshape(-1, 3).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    args = parser.parse_args()

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False, max_collision_pairs=40),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.0), camera_lookat=(0.0, 0.0, 0.3), camera_fov=40
        ),
        show_viewer=args.viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0.0, 0.0, 0.4)),
        # Keep the tendon from colliding with the leg (particles would be pushed
        # out and jerk the robot). The plane still collides with the tendon.
        material=gs.materials.Rigid(needs_coup=False),
        name="go2",
    )

    make_ribbon("/tmp/opencode/go2_tendon.obj")
    tendon = scene.add_entity(
        gs.morphs.Mesh(file="/tmp/opencode/go2_tendon.obj"),
        material=gs.materials.PBD.Cloth(
            rho=4.0,  # kg/m^2 (2D material)
            stretch_compliance=1e-7,  # m/N; smaller = stiffer line
            stretch_relaxation=0.3,
            bending_compliance=1e-5,
            air_resistance=1e-3,
        ),
        name="tendon",
    )

    scene.build(n_envs=1)

    # ------------------------------------------------------------------
    # Route the tendon from the hip, over the knee, down to the calf, then
    # pin its two ends to the links it should ride on.
    # ------------------------------------------------------------------
    p_thigh = robot.get_link("FL_thigh").get_pos().cpu().numpy().reshape(3)
    p_calf = robot.get_link("FL_calf").get_pos().cpu().numpy().reshape(3)

    n_nodes = tendon.n_particles // 2
    w = np.linspace(0.0, 1.0, n_nodes)
    route = p_thigh[None] + w[:, None] * (p_calf - p_thigh)[None]
    verts = warp_along(route, np.array([0.0, 3e-3, 0.0]))
    tendon.set_particles_pos(verts)
    tendon.set_particles_vel(np.zeros_like(verts))

    pos = tendon.get_particles_pos().cpu().numpy().reshape(-1, 3)
    top = int(np.argmin(np.linalg.norm(pos - p_thigh, axis=1)))
    bottom = int(np.argmin(np.linalg.norm(pos - p_calf, axis=1)))
    # Pin a couple of particles per anchor so the ribbon cannot spin freely.
    tendon.fix_particles_to_link(robot.get_link("FL_thigh").idx, [top, top ^ 1])
    tendon.fix_particles_to_link(robot.get_link("FL_calf").idx, [bottom, bottom ^ 1])
    print(f"tendon: {tendon.n_particles} particles, anchors {top}->FL_thigh, {bottom}->FL_calf")

    # ------------------------------------------------------------------
    # Drive the leg and let the soft line follow.
    # ------------------------------------------------------------------
    q_target = torch.zeros((1, 12), dtype=gs.tc_float, device=gs.device)
    for step in range(600):
        if step % 60 == 0:
            q_target[0, 2] = 0.5 * np.sin(step / 20)  # sweep the knees a bit
        robot.control_dofs_position(q_target, slice(6, 18))
        scene.step()

    print("done; final tendon bbox:", np.ptp(tendon.get_particles_pos().cpu().numpy().reshape(-1, 3), axis=0))


def tendon_torque(jacobian, tension):
    """Standard sim-to-real path: tau = J^T * tension.

    Compute the scalar tendon length L(q) from the joint angles and routing
    (moment arms / pulley geometry), then drive the joint with this torque
    instead of (or on top of) the PD controller. This is how the tension is
    transmitted back to the leg, something particle pinning does not do.
    """
    return jacobian.T @ tension


if __name__ == "__main__":
    main()
