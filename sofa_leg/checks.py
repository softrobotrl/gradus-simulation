"""Physics checks for the SOFA rod, against textbook beam formulas.

  rod sag     a rod-only cantilever (0.6 m, the leg's rod) sagging under its own weight, against q L^4 / (8 E I),
              at several mesh resolutions. Linear hexahedra are too stiff in bending ("shear locking") unless the
              elements are short, so this shows how much of the leg's stiffness is mesh error at each resolution.
  disk hang   a rigid 1 kg disk clamped at the end of a 0.2 m cantilever, built with the same scene code as the leg
              (sofa_leg.scene.build_leg). Its sag must match P L^3 / (3 E I) + its moment term, scaled by the mesh's
              own stiffness error from the point-load cantilever. Proves the disk-rod coupling works both ways:
              with one-way coupling the disk would fall freely.

Usage:
  python -m sofa_leg.checks
"""

import argparse
import math
import sys
import time

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

from sofa_leg.rodmesh import axial_layers, rod_mesh
from sofa_leg.scene import build_leg, load_plugins
from sofa_leg.spec import Disk, LegSpec, RodParams

RESOLUTIONS = ((3, 0.005), (3, 0.0025), (3, 0.0015), (3, 0.001), (4, 0.0015))
SAG_LENGTH = 0.6
HANG_LENGTH, HANG_MASS, HANG_THICKNESS = 0.2, 1.0, 0.0196
TOLERANCE = 0.05


def _static_rod(rod, length, n_side, element, point_load=None):
    """Static cantilever along -z, clamped at z=0, gravity (or a tip load) along +x. Returns the tip's x."""
    nodes, hexes, density_scale = rod_mesh(rod.radius, axial_layers([0, length], element), n_side,
                                           axis=(0, 0, -1))
    per_layer = n_side ** 2
    root = Sofa.Core.Node("root")
    root.gravity = [0.0 if point_load else 9.81, 0, 0]
    root.dt = 1.0
    load_plugins()
    root.addObject("DefaultAnimationLoop")
    node = root.addChild("rod")
    node.addObject("NewtonRaphsonSolver", name="newton", maxNbIterationsNewton=20,
                   absoluteResidualStoppingThreshold=1e-8, warnWhenLineSearchFails=False, warnWhenDiverge=False)
    node.addObject("StaticSolver", newtonSolver="@newton")
    node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    node.addObject("HexahedronSetTopologyContainer", name="topology", position=nodes.tolist(),
                   hexahedra=hexes.tolist())
    node.addObject("MechanicalObject", name="dofs", position=nodes.tolist())
    node.addObject("MeshMatrixMass", massDensity=rod.density * density_scale, topology="@topology")
    node.addObject("HexahedronFEMForceField", youngModulus=rod.youngs_modulus, poissonRatio=rod.poisson_ratio,
                   method="large", topology="@topology")
    node.addObject("FixedProjectiveConstraint", indices=list(range(per_layer)))
    if point_load:
        tip = list(range(len(nodes) - per_layer, len(nodes)))
        node.addObject("ConstantForceField", indices=tip, forces=[[point_load / per_layer, 0, 0]] * per_layer)
    Sofa.Simulation.init(root)
    for _ in range(3):
        Sofa.Simulation.animate(root, 1.0)
    pos = np.asarray(node.dofs.position.value)
    return float(pos[-per_layer:, 0].mean() - nodes[-per_layer:, 0].mean()), len(nodes)


def rod_sag(rod):
    q = rod.density * math.pi * rod.radius ** 2 * 9.81
    expected = q * SAG_LENGTH ** 4 / (8 * rod.bending_stiffness)
    rows = []
    for n_side, element in RESOLUTIONS:
        start = time.perf_counter()
        sag, n_nodes = _static_rod(rod, SAG_LENGTH, n_side, element)
        rows.append((n_side, element, n_nodes, sag, sag / expected, time.perf_counter() - start))
    return expected, rows


def point_load_ratio(rod, length, element):
    """The mesh's own bending stiffness error for a tip load: FEM deflection / beam-theory deflection."""
    load = 1.0
    sag, _ = _static_rod(rod, length, rod.n_side, element, point_load=load)
    return sag / (load * length ** 3 / (3 * rod.bending_stiffness))


def _hang_spec(rod):
    """Two disks on a vertical rod: a fixed clamp at the top, a free 1 kg disk at the bottom."""
    def disk(name, mass, centre, thickness):
        r, t = 0.03, thickness
        inertia = np.array([mass * (3 * r ** 2 + t ** 2) / 12] * 2 + [mass * r ** 2 / 2])
        return Disk(name, np.array([0, 0, -centre]), np.array([0, 0, 0, 1.0]), np.eye(3), mass, inertia,
                    (centre - t / 2, centre + t / 2), np.zeros((0, 3)), np.zeros((0, 3), dtype=int))

    clamp = disk("clamp", 0.2, -0.01, 0.02)
    hang = disk("hang", HANG_MASS, HANG_LENGTH + HANG_THICKNESS / 2, HANG_THICKNESS)
    return LegSpec(xml="(synthetic)", xml_sha256="", disks=[clamp, hang], tendons=[],
                   axis_origin=np.zeros(3), axis=np.array([0, 0, -1.0]), rod_span=(-0.02, HANG_LENGTH + HANG_THICKNESS),
                   tip_disk=1, tip_local=np.zeros(3), gravity=np.array([9.81, 0, 0]), timestep=0.002,
                   joint_stiffness=1.0, joint_damping=0.01, joint_frictionloss=0.0, rod=rod)


def disk_hang(rod, seconds=3.0, dt=0.002):
    """Settle the hanging disk with the leg's own scene code. Returns (measured, beam theory, mesh ratio)."""
    light = RodParams(**{**rod.__dict__, "density": 1e-3})   # the disk's weight only
    spec = _hang_spec(light)
    root = Sofa.Core.Node("root")
    handle = build_leg(root, spec, dt=dt, rayleigh=0.01)
    Sofa.Simulation.init(root)
    for _ in range(round(seconds / dt)):
        Sofa.Simulation.animate(root, dt)
    frame = np.asarray(handle.disks.dofs.position.value)[1]
    velocity = np.asarray(handle.disks.dofs.velocity.value)[1]
    measured = float(frame[0])

    ei, length, load = rod.bending_stiffness, HANG_LENGTH, HANG_MASS * 9.81
    moment = load * HANG_THICKNESS / 2           # the disk's weight acts half a thickness past the rod's free end
    slope = load * length ** 2 / (2 * ei) + moment * length / ei
    end = load * length ** 3 / (3 * ei) + moment * length ** 2 / (2 * ei)
    expected = end + slope * HANG_THICKNESS / 2
    return measured, expected, point_load_ratio(light, length, rod.free_element), float(np.abs(velocity).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    rod = RodParams()
    ok = True

    expected, rows = rod_sag(rod)
    print(f"rod sag: {SAG_LENGTH} m cantilever, d = {rod.diameter * 1e3:.4g} mm, E = {rod.youngs_modulus / 1e9:g} GPa,"
          f" own weight; beam theory {expected * 1e3:.3f} mm")
    print("  nodes across  element  nodes   sag (mm)  FEM / beam   time (s)")
    for n_side, element, n_nodes, sag, ratio, seconds in rows:
        mark = "  <- leg default" if (n_side, element) == (rod.n_side, rod.free_element) else ""
        print(f"  {n_side:>12}  {element * 1e3:5.2f} mm  {n_nodes:5d}  {sag * 1e3:8.3f}  {ratio:10.3f}  {seconds:8.2f}{mark}")
    default = [r for r in rows if (r[0], r[1]) == (rod.n_side, rod.free_element)]
    if default:
        stiff = 1 / default[0][4] - 1
        print(f"  the leg's mesh bends {stiff:.1%} stiffer than beam theory (mesh error, shrinks as elements shorten)")

    measured, expected, ratio, speed = disk_hang(rod)
    held = measured > 0 and speed < 1e-3
    match = abs(measured / (expected * ratio) - 1) <= TOLERANCE
    ok &= held and match
    print(f"disk hang: {HANG_MASS} kg disk clamped {HANG_LENGTH} m down a cantilever, gravity sideways")
    print(f"  {'PASS' if held and match else 'FAIL'}  sag {measured * 1e3:.3f} mm; beam theory {expected * 1e3:.3f} mm;"
          f" x mesh ratio {ratio:.3f} (point-load cantilever, same mesh) = {expected * ratio * 1e3:.3f} mm;"
          f" disk speed after settling {speed:.1e} m/s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
