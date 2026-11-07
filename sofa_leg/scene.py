"""Build the leg as a SOFA scene graph: an FEM fibreglass rod, rigid disks clamped onto it, and 8 tendon servos.

Scene graph (names in quotes are the node names):

    root                          gravity, FreeMotionAnimationLoop, BlockGaussSeidelConstraintSolver
    └─ "leg"                      EulerImplicitSolver (Rayleigh damping), SparseLDLSolver, GenericConstraintCorrection
       ├─ "free"                  MechanicalObject Vec3d: the rod nodes no disk holds (independent DOFs)
       ├─ "disks"                 MechanicalObject Rigid3d: one frame per disk (independent DOFs), per-disk UniformMass,
       │  │                       the hip mount fixed
       │  ├─ "clamped"            rod nodes held by a disk, RigidMapping (they move with their disk)
       │  ├─ "holes"              every tendon hole, RigidMapping; one CableConstraint per tendon lives here
       │  └─ "visual_<i>"         the disk's STL, RigidMapping (only with visual=True)
       └─ "rod"                   all rod nodes in mesh order, SubsetMultiMapping of "free" + "clamped",
                                  hexahedral FEM (HexahedronFEMForceField, corotational) and the rod's mass

The disks are true rigid bodies (6 DOF each) and the rod nodes inside a disk are mapped onto it, so the rod's elastic
forces reach the disk and the disk's inertia and tendon loads reach the rod: a two-way coupling with no penalty
springs. It is the same "rigidification" pattern as STLIB's stlib3.physics.mixedmaterial.Rigidify, written out here
so each disk can have its own mass and inertia.

Tendons are SoftRobots CableConstraints in force mode, threaded through their holes in order (hip hole first,
hasPullPoint=False), so a cable's length is the same sum of hole-to-hole distances MuJoCo uses for its spatial tendon,
and it slides through the holes without friction, like MuJoCo's. Each step ServoController sets every cable's tension
to MuJoCo's position-servo law (see servo_tension). Measured on SOFA v26.06: in force mode, CableConstraint's `value`
is an impulse per time step (tension x dt), not a force.
"""

import numpy as np

import Sofa
import Sofa.Core

from sofa_leg.rodmesh import axial_layers, rod_mesh

PLUGINS = ("Sofa.Component", "SoftRobots")
VISUAL_PLUGINS = ("Sofa.GL.Component",)


def servo_tension(length, length_rate, ctrl, kp, kv, fmax):
    """MuJoCo's <position> servo on a tendon, as a tension (>= 0).

    MuJoCo: force = kp (ctrl - L) - kv dL/dt, clamped to forcerange [-fmax, 0]; the tendon pulls with -force.
    """
    return np.clip(kp * (np.asarray(length) - ctrl) + kv * np.asarray(length_rate), 0.0, fmax)


def quat_rotate(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) in x y z w order."""
    q = np.asarray(q)
    u, w = q[..., :3], q[..., 3:4]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rod_layout(spec):
    """Rod mesh plus which disk (if any) holds each node.

    Returns (nodes, hexahedra, density_scale, owner) where owner[i] is the disk index holding node i, or -1.
    """
    rod = spec.rod
    start, end = spec.rod_span
    middles = [0.5 * (d.axial[0] + d.axial[1]) for d in spec.disks]
    breaks = [start, end] + [a for d in spec.disks for a in d.axial] + middles
    points = np.unique(np.round(np.clip(breaks, start, end), 9))

    def clamped(a, b):
        mid = 0.5 * (a + b)
        holds = [i for i, d in enumerate(spec.disks) if d.axial[0] - 1e-9 <= mid <= d.axial[1] + 1e-9]
        return bool(holds) and (rod.grip == "thickness" or holds == [0])

    layers = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        layers.extend(axial_layers([a, b], rod.clamped_element if clamped(a, b) else rod.free_element)[1:])
    layers = np.array(layers)

    nodes, hexes, density_scale = rod_mesh(rod.radius, layers, rod.n_side, spec.axis_origin, spec.axis)
    per_layer = rod.n_side ** 2
    layer_owner = np.full(len(layers), -1)
    for i, d in enumerate(spec.disks):
        if rod.grip == "thickness" or i == 0:  # the hip mount is a fixed clamp in both modes
            inside = (layers >= d.axial[0] - 1e-9) & (layers <= d.axial[1] + 1e-9)
        else:
            inside = np.zeros(len(layers), bool)
            inside[np.argmin(np.abs(layers - middles[i]))] = True
        layer_owner[inside & (layer_owner < 0)] = i
    owner = np.repeat(layer_owner, per_layer)
    return nodes, hexes, density_scale, owner


class ServoController(Sofa.Core.Controller):
    """Sets each cable's tension from MuJoCo's servo law, from the current hole positions and velocities.

    Hole positions and velocities are computed from the disk frames directly (not read from the mapped "holes"
    object), so they are exact even right after the state has been overwritten by `SofaLeg.set_state`.
    """

    def __init__(self, leg, spec, dt, **kwargs):
        super().__init__(**kwargs)
        self.leg, self.spec, self.dt = leg, spec, dt
        self.ctrl = np.array([t.ctrl_high for t in spec.tendons])
        self.tension = np.zeros(len(spec.tendons))
        self.length = np.array([t.rest_length for t in spec.tendons])
        self.length_rate = np.zeros(len(spec.tendons))

    def update(self):
        frames = np.asarray(self.leg.disks.dofs.position.value)
        velocities = np.asarray(self.leg.disks.dofs.velocity.value)
        for k, t in enumerate(self.spec.tendons):
            q = frames[t.disks]
            r = quat_rotate(q[:, 3:], t.local)
            p = q[:, :3] + r
            v = velocities[t.disks, :3] + np.cross(velocities[t.disks, 3:], r)
            seg = np.diff(p, axis=0)
            seg_len = np.linalg.norm(seg, axis=1)
            unit = seg / np.maximum(seg_len, 1e-12)[:, None]
            self.length[k] = seg_len.sum()
            self.length_rate[k] = np.einsum("ij,ij->", unit, np.diff(v, axis=0))
        self.tension = servo_tension(self.length, self.length_rate, self.ctrl,
                                     np.array([t.kp for t in self.spec.tendons]),
                                     np.array([t.kv for t in self.spec.tendons]),
                                     np.array([t.fmax for t in self.spec.tendons]))
        for cable, tension in zip(self.leg.cables, self.tension):
            cable.value = [float(tension) * self.dt]

    def onAnimateBeginEvent(self, event):
        self.update()


def load_plugins(visual=False):
    """Load SOFA's components (imported directly: a RequiredPlugin for the Sofa.Component meta-plugin only warns)."""
    import SofaRuntime

    for name in PLUGINS + (VISUAL_PLUGINS if visual else ()):
        SofaRuntime.importPlugin(name)


def build_leg(root, spec, dt=0.001, rayleigh=None, visual=False, linear_solver="SparseLDLSolver"):
    """Add the leg under `root`. Returns a handle with the nodes and objects SofaLeg needs.

    rayleigh: stiffness-proportional damping coefficient beta (s). Default: MuJoCo's joint damping / joint stiffness,
    so each bending mode gets the damping ratio it has in MuJoCo (zeta = beta * omega / 2 in both models).
    """
    rod = spec.rod
    if rayleigh is None:
        rayleigh = spec.joint_damping / spec.joint_stiffness
    root.gravity = spec.gravity.tolist()
    root.dt = dt
    load_plugins(visual)
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("BlockGaussSeidelConstraintSolver", maxIterations=1000, tolerance=1e-10)
    if visual:
        root.addObject("VisualStyle", displayFlags="showVisualModels")
        root.addObject("LightManager")
        root.addObject("DirectionalLight", name="light_1", direction=[1, 1, 1])
        root.addObject("DirectionalLight", name="light_2", direction=[-1, -1, 1])

    leg = root.addChild("leg")
    leg.addObject("EulerImplicitSolver", rayleighStiffness=rayleigh, rayleighMass=0.0)
    if linear_solver == "CGLinearSolver":
        leg.addObject("CGLinearSolver", iterations=200, tolerance=1e-12, threshold=1e-14)
    else:
        leg.addObject(linear_solver, template="CompressedRowSparseMatrixd")
    leg.addObject("GenericConstraintCorrection")

    nodes, hexes, density_scale, owner = rod_layout(spec)
    free_idx = np.flatnonzero(owner < 0)
    held_idx = np.flatnonzero(owner >= 0)

    free = leg.addChild("free")
    free.addObject("MechanicalObject", name="dofs", template="Vec3d", position=nodes[free_idx].tolist())

    disks = leg.addChild("disks")
    disks.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                    position=[d.pos.tolist() + d.quat.tolist() for d in spec.disks])
    disks.addObject("FixedProjectiveConstraint", indices=[0])
    for i, d in enumerate(spec.disks):
        # SOFA allows one mass per node, so each disk's mass sits on a one-frame copy of it (SubsetMapping).
        # RigidMass takes the inertia per unit mass (checked on v26.06: torque / (m * inertiaMatrix)).
        body = disks.addChild(f"body_{i}")
        body.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=[d.pos.tolist() + d.quat.tolist()])
        body.addObject("UniformMass", vertexMass=[d.mass, 1.0, np.diag(d.inertia / d.mass).ravel().tolist()])
        body.addObject("SubsetMapping", indices=[i], handleTopologyChange=False)

    clamped = disks.addChild("clamped")
    clamped.addObject("MechanicalObject", name="dofs", template="Vec3d", position=nodes[held_idx].tolist())
    clamped.addObject("RigidMapping", globalToLocalCoords=True, rigidIndexPerPoint=owner[held_idx].tolist())

    cables, start, holes = [], 0, None
    if spec.tendons:
        holes = disks.addChild("holes")
        hole_world = np.concatenate([t.world for t in spec.tendons])
        hole_disk = np.concatenate([t.disks for t in spec.tendons])
        holes.addObject("MechanicalObject", name="dofs", template="Vec3d", position=hole_world.tolist())
        holes.addObject("RigidMapping", globalToLocalCoords=True, rigidIndexPerPoint=hole_disk.tolist())
    for t in spec.tendons:
        cables.append(holes.addObject("CableConstraint", name=t.name, template="Vec3d",
                                      indices=list(range(start, start + len(t.disks))), hasPullPoint=False,
                                      valueType="force", value=0.0, drawPoints=False))
        start += len(t.disks)

    rod_node = free.addChild("rod")
    clamped.addChild(rod_node)
    rod_node.addObject("HexahedronSetTopologyContainer", name="topology", position=nodes.tolist(),
                       hexahedra=hexes.tolist())
    rod_node.addObject("MechanicalObject", name="dofs", template="Vec3d", position=nodes.tolist())
    pairs = np.zeros((len(nodes), 2), dtype=int)
    pairs[free_idx] = np.stack([np.zeros(len(free_idx), int), np.arange(len(free_idx))], 1)
    pairs[held_idx] = np.stack([np.ones(len(held_idx), int), np.arange(len(held_idx))], 1)
    rod_node.addObject("SubsetMultiMapping", template="Vec3d,Vec3d",
                       input=[free.dofs.getLinkPath(), clamped.dofs.getLinkPath()],
                       output=rod_node.dofs.getLinkPath(), indexPairs=pairs.ravel().tolist())
    rod_node.addObject("MeshMatrixMass", name="mass", massDensity=rod.density * density_scale, topology="@topology")
    rod_node.addObject("HexahedronFEMForceField", name="fem", youngModulus=rod.youngs_modulus,
                       poissonRatio=rod.poisson_ratio, method="large", topology="@topology")

    if visual:
        for i, d in enumerate(spec.disks):
            v = disks.addChild(f"visual_{i}")
            v.addObject("OglModel", name="model", position=d.vertices.tolist(), triangles=d.faces.tolist(),
                        color=[0.85, 0.55, 0.2, 1.0] if 0 < i < len(spec.disks) - 1 else [0.5, 0.5, 0.55, 1.0])
            v.addObject("RigidMapping", index=i, globalToLocalCoords=False)
        surface = rod_node.addChild("surface")
        surface.addObject("QuadSetTopologyContainer", name="quads")
        surface.addObject("QuadSetTopologyModifier")
        surface.addObject("Hexa2QuadTopologicalMapping", input="@../topology", output="@quads")
        surface.addObject("OglModel", name="model", topology="@quads", color=[0.2, 0.6, 0.3, 1.0])
        surface.addObject("IdentityMapping")

    handle = type("LegHandle", (), {})()
    handle.root, handle.node, handle.free, handle.disks, handle.rod = root, leg, free, disks, rod_node
    handle.holes, handle.cables, handle.nodes, handle.owner = holes, cables, nodes, owner
    handle.rayleigh, handle.dt = rayleigh, dt
    handle.servo = root.addObject(ServoController(handle, spec, dt, name="servo"))
    return handle
