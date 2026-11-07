"""Everything the SOFA leg needs, read from a merged leg.xml through MuJoCo. Needs mujoco and numpy, not SOFA.

The SOFA scene is built from this spec, so the geometry, masses, tendon routes and servo settings are exactly the
ones MuJoCo compiles from Daniel's leg.xml. Nothing is re-derived from the CAD.

Frames: every disk is placed at its MuJoCo *inertial* frame (centre of mass, principal axes) at qpos0, so its inertia
is the diagonal `body_inertia`. Hole points, mesh vertices and the tip are stored in that frame too.
Quaternions are SOFA's order, (x, y, z, w).

What leg.xml does not contain is the rod itself (MuJoCo only has its stiffness, k = EI / pitch). The rod's
diameter and Young's modulus default to the values Daniel's pipeline used to compute that k
(mujoco_to_mjcf/config.json: d = 7.9375 mm, E = 30 GPa). Poisson's ratio and density are assumptions.
"""

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from genesis_leg.leg_model import find_tip, read_routes

GRIPS = ("thickness", "plane")


@dataclass
class RodParams:
    """The fibreglass backbone and how it is meshed.

    grip: how a disk holds the rod.
      "thickness" - every rod node inside the disk moves with the disk (clamped over the disk's full thickness,
                    25 mm of free rod between disks). The physical default.
      "plane"     - only the node layer at the disk's mid-plane moves with it (free rod = the full 44.6 mm pitch).
                    This is the assumption behind MuJoCo's k = EI / pitch.
    """

    diameter: float = 0.0079375      # m, mujoco_to_mjcf/config.json rod_diameter_m
    youngs_modulus: float = 30e9     # Pa, mujoco_to_mjcf/config.json rod_youngs_modulus_pa
    poisson_ratio: float = 0.3       # assumed (typical for glass-fibre/epoxy, 0.25-0.3)
    density: float = 1900.0          # kg/m^3, assumed (pultruded fibreglass is 1800-2000); MuJoCo's leg has no rod mass
    n_side: int = 3                  # cross-section nodes per side (3 -> 2 x 2 hexahedra across the rod)
    free_element: float = 0.0015     # m, longest element in rod that bends (between disks)
    clamped_element: float = 0.005   # m, longest element inside a disk (rigid there, so coarse is fine)
    grip: str = "thickness"

    def __post_init__(self):
        if self.grip not in GRIPS:
            raise ValueError(f"grip must be one of {GRIPS}, got {self.grip!r}")

    @property
    def radius(self):
        return self.diameter / 2

    @property
    def second_moment(self):
        return math.pi * self.diameter ** 4 / 64

    @property
    def bending_stiffness(self):
        return self.youngs_modulus * self.second_moment


@dataclass
class Disk:
    name: str
    pos: np.ndarray            # (3,) world, centre of mass
    quat: np.ndarray           # (4,) world, x y z w (principal axes)
    rot: np.ndarray            # (3, 3) world-from-local
    mass: float
    inertia: np.ndarray        # (3,) principal moments, kg m^2
    axial: tuple               # (start, end) of the disk along the rod axis, m
    vertices: np.ndarray       # (V, 3) visual mesh, local frame
    faces: np.ndarray          # (F, 3)

    def to_world(self, local):
        return self.pos + np.asarray(local) @ self.rot.T

    def to_local(self, world):
        return (np.asarray(world) - self.pos) @ self.rot


@dataclass
class Tendon:
    name: str
    disks: list                # disk index of each hole, hip first
    local: np.ndarray          # (H, 3) hole positions in each disk's local frame
    world: np.ndarray          # (H, 3) at qpos0
    rest_length: float         # MuJoCo ten_length at qpos0
    kp: float
    kv: float
    fmax: float
    ctrl_low: float
    ctrl_high: float


@dataclass
class LegSpec:
    xml: str
    xml_sha256: str
    disks: list
    tendons: list
    axis_origin: np.ndarray    # a point on the rod axis (the hip body's origin)
    axis: np.ndarray           # unit vector from hip to foot
    rod_span: tuple            # (start, end) along the axis
    tip_disk: int
    tip_local: np.ndarray
    gravity: np.ndarray
    timestep: float            # MuJoCo's, for reference
    joint_stiffness: float     # MuJoCo ball-joint stiffness, N m / rad
    joint_damping: float       # MuJoCo ball-joint damping, N m s / rad
    joint_frictionloss: float
    rod: RodParams = field(default_factory=RodParams)

    @property
    def disk_mass(self):
        return sum(d.mass for d in self.disks)

    def axial_coordinate(self, world):
        return (np.asarray(world) - self.axis_origin) @ self.axis

    def tip_world(self):
        return self.disks[self.tip_disk].to_world(self.tip_local)

    def summary(self):
        return {
            "xml": self.xml, "xml_sha256": self.xml_sha256, "n_disks": len(self.disks),
            "tendons": [t.name for t in self.tendons], "disk_mass_kg": self.disk_mass,
            "rod": asdict(self.rod), "rod_length_m": self.rod_span[1] - self.rod_span[0],
            "joint_stiffness": self.joint_stiffness, "joint_damping": self.joint_damping,
        }


def _xyzw(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).ravel())
    return np.array([q[1], q[2], q[3], q[0]])


def _body_mesh(model, data, body, world_to_local):
    """Visual mesh geoms of one body, merged, in the given local frame."""
    vertices, faces = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            continue  # the visual copies; collision copies of the same mesh are skipped
        mesh = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        world = data.geom_xpos[g] + model.mesh_vert[v0:v0 + nv] @ data.geom_xmat[g].reshape(3, 3).T
        faces.append(model.mesh_face[f0:f0 + nf] + sum(len(v) for v in vertices))
        vertices.append(world_to_local(world))
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.concatenate(vertices), np.concatenate(faces).astype(int)


def leg_spec(xml, rod=None):
    """Compile leg.xml with MuJoCo and return a LegSpec at qpos0."""
    xml = Path(xml)
    rod = rod or RodParams()
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bodies = list(range(1, model.nbody))
    if model.body_parentid[bodies[0]] != 0 or model.body_jntnum[bodies[0]] != 0:
        raise ValueError("expected the first body to be the fixed hip mount")
    for b in bodies[1:]:
        if model.body_jntnum[b] != 1 or model.jnt_type[model.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_BALL:
            raise ValueError(f"{model.body(b).name}: expected exactly one ball joint (a chain of disks)")

    origin = data.xpos[bodies[0]].copy()
    axis = data.xpos[bodies[-1]] - origin
    axis /= np.linalg.norm(axis)
    off_axis = max(np.linalg.norm(np.cross(data.xpos[b] - origin, axis)) for b in bodies)
    if off_axis > 1e-5:
        raise ValueError(f"the disks are not on one straight line at qpos0 (off by {off_axis * 1e3:.3f} mm)")

    disks = []
    for b in bodies:
        rot = data.ximat[b].reshape(3, 3).copy()
        pos = data.xipos[b].copy()
        vertices, faces = _body_mesh(model, data, b, lambda w, p=pos, r=rot: (w - p) @ r)
        if len(vertices) == 0:
            raise ValueError(f"{model.body(b).name}: no visual mesh to take the disk's thickness from")
        along = (vertices @ rot.T + pos - origin) @ axis
        disks.append(Disk(model.body(b).name, pos, _xyzw(rot), rot, float(model.body_mass[b]),
                          model.body_inertia[b].copy(), (float(along.min()), float(along.max())), vertices, faces))

    index = {d.name: i for i, d in enumerate(disks)}
    routes = read_routes(model)
    tendons = []
    for t, route in enumerate(routes):
        hole_disks = [index[name] for name in route["bodies"]]
        world = np.array([data.xpos[model.body(name).id] + data.xmat[model.body(name).id].reshape(3, 3) @ offset
                          for name, offset in zip(route["bodies"], route["offsets"])])
        local = np.array([disks[i].to_local(w) for i, w in zip(hole_disks, world)])
        tendon_id = model.actuator_trnid[t, 0]
        tendons.append(Tendon(route["name"], hole_disks, local, world, float(data.ten_length[tendon_id]),
                              route["kp"], route["kv"], route["fmax"], route["ctrl_low"], route["ctrl_high"]))
        if hole_disks[0] != 0:
            raise ValueError(f"{route['name']}: expected the route to start at the hip mount")

    tip_body, tip_offset = find_tip(model, routes)
    b = model.body(tip_body).id
    tip_world = data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.asarray(tip_offset)
    tip_disk = index[tip_body]

    joint = model.body_jntadr[bodies[1]]
    dof = model.jnt_dofadr[joint]
    span = (min(d.axial[0] for d in disks), max(d.axial[1] for d in disks))
    return LegSpec(
        xml=str(xml), xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(), disks=disks, tendons=tendons,
        axis_origin=origin, axis=axis, rod_span=span, tip_disk=tip_disk,
        tip_local=disks[tip_disk].to_local(tip_world), gravity=model.opt.gravity.copy(),
        timestep=float(model.opt.timestep), joint_stiffness=float(model.jnt_stiffness[joint]),
        joint_damping=float(model.dof_damping[dof]), joint_frictionloss=float(model.dof_frictionloss[dof]), rod=rod)
