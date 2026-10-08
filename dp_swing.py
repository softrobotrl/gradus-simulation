"""Free-swing test: no control, see how long the elbow keeps swinging.

  python free_swing_test.py                  # use damping/friction as in the URDF
  python free_swing_test.py --damping 0      # override damping on all joints
  python free_swing_test.py --view           # watch it

Prints the elbow's peak angle in each 0.5 s window. A nearly frictionless pendulum
decays slowly (peaks shrink a few percent per window). If the peak collapses within
a window or two, damping/friction is still too high (edit the URDF <dynamics> tags).
"""
import argparse

import numpy as np
import torch
import genesis as gs

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", default="robot.urdf")
ap.add_argument("--names", nargs=3, default=["cart", "shoulder_continuous", "elbow_continuous"])
ap.add_argument("--damping", type=float, default=None)
ap.add_argument("--angle", type=float, default=0.8, help="initial elbow offset (rad)")
ap.add_argument("--hanging", type=float, default=0.0, help="shoulder angle for 'hanging down'")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--view", action="store_true")
args = ap.parse_args()

gs.init(backend=gs.cpu, logging_level="warning")
dt = 0.01
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=2),
    rigid_options=gs.options.RigidOptions(enable_self_collision=False),
    show_viewer=args.view,
)
robot = scene.add_entity(gs.morphs.URDF(file=args.urdf, fixed=True))
scene.build()
for l in robot.links:
    print(l.name, l.get_mass())

def dof(name):
    j = robot.get_joint(name)
    idx = getattr(j, "dofs_idx_local", None)
    return idx[0] if idx is not None else j.dof_idx_local


dofs = [dof(n) for n in args.names]
if args.damping is not None:
    robot.set_dofs_damping(torch.full((3,), args.damping), dofs)

robot.set_dofs_position(torch.tensor([0.0, args.hanging, args.angle]), dofs)

window = int(0.5 / dt)
trace = []
for step in range(int(args.seconds / dt)):
    scene.step()
    q = robot.get_dofs_position(dofs).cpu().numpy().reshape(-1)
    trace.append(q[2])
    if (step + 1) % window == 0:
        w = np.array(trace[-window:])
        print(f"t={(step + 1) * dt:5.1f}s  elbow peak |q|={np.abs(w).max():.3f}  "
              f"shoulder={q[1]:.3f}")