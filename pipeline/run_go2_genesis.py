"""Unitree Go2 standing on random terrain in the Genesis viewer.

Usage: python -m pipeline.run_go2_genesis --terrain mixed --seed 3 --difficulty 0.5 --grid 4 --backend cpu

The terrain comes from pipeline/terrain.py, so a seed gives the same ground in every simulator that uses it. Go2 (the
URDF bundled with Genesis, a stand-in until the Gradus quadruped model exists) starts on a flat tile and holds its home
pose with PD control; there is no policy yet. Left-drag it with the mouse to pull it onto the terrain. Close the window
to stop. The first build takes about 40 s on a laptop CPU while Genesis compiles; later ones take a few seconds.
"""

import argparse

import genesis as gs
import numpy as np

from genesis_leg.train import initialize
from pipeline.terrain import CELL, KINDS, Terrain, generate

# Home pose and gains from the Genesis Go2 locomotion example.
HOME_POSE = {  # rad
    "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}
KP = 20.0  # N*m/rad
KD = 0.5  # N*m*s/rad
SPAWN_HEIGHT = 0.35  # m, base above the terrain: the feet just clear it in the home pose


def new_scene(terrain: Terrain, show_viewer: bool = False) -> gs.Scene:
    x, y = terrain.spawn_xy
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        viewer_options=gs.options.ViewerOptions(camera_pos=(x + 2.5, y - 2.5, 2.0), camera_lookat=(x, y, 0.2)),
        show_viewer=show_viewer,
    )


def add_terrain(scene: gs.Scene, terrain: Terrain):
    # Genesis indexes height_field[x, y] from the corner at pos; Terrain.heights is [y, x] and centred on the origin.
    return scene.add_entity(gs.morphs.Terrain(height_field=terrain.heights.T, horizontal_scale=CELL, vertical_scale=1.0,
                                              pos=(-terrain.size / 2, -terrain.size / 2, 0.0)))


def add_go2(scene: gs.Scene, terrain: Terrain):
    x, y = terrain.spawn_xy
    z = terrain.max_height(x, y) + SPAWN_HEIGHT
    return scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(x, y, z)))


def hold_home_pose(robot) -> None:
    """Start in the home pose and hold it with PD control. Call after scene.build()."""
    dofs = [robot.get_joint(name).dofs_idx_local[0] for name in HOME_POSE]
    target = np.array(list(HOME_POSE.values()))
    robot.set_dofs_kp(np.full(len(dofs), KP), dofs)
    robot.set_dofs_kv(np.full(len(dofs), KD), dofs)
    robot.set_dofs_position(target, dofs)
    robot.control_dofs_position(target, dofs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrain", choices=("mixed", *KINDS, "flat"), default="mixed")
    parser.add_argument("--seed", type=int, help="omit for a random seed (printed so the run can be repeated)")
    parser.add_argument("--difficulty", type=float, default=0.5,
                        help="0..1: max per-tile bump height, slope angle, block height")
    parser.add_argument("--grid", type=int, default=4, help="N for an N x N grid of 2 m tiles")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(2**31))
    print(f"rerun with: python -m pipeline.run_go2_genesis --terrain {args.terrain} --seed {seed} "
          f"--difficulty {args.difficulty} --grid {args.grid} --backend {args.backend}", flush=True)

    initialize(args.backend, seed)
    terrain = generate(args.terrain, seed, args.difficulty, args.grid)
    scene = new_scene(terrain, show_viewer=True)
    add_terrain(scene, terrain)
    robot = add_go2(scene, terrain)
    scene.viewer.add_plugin(gs.vis.viewer_plugins.MouseInteractionPlugin())
    scene.build()
    hold_home_pose(robot)
    while scene.viewer.is_alive():
        scene.step()


if __name__ == "__main__":
    main()
