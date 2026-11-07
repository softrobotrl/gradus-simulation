"""The merged Onshape leg in Genesis: load a leg.xml, drive its tendon servos, read back its state.

    from genesis_leg.leg import GenesisLeg, init_genesis
    init_genesis("auto")
    leg = GenesisLeg("urdf/gradus/sim/leg.xml", num_envs=4)
    leg.step(leg.pull_to_ctrl(pull))    # pull: (num_envs, tendons) in [0, 1]; 0 = rest length, 1 = full stroke

ctrl is the target tendon length in metres, as in MuJoCo. There is no RL task here: see genesis_leg/README.md.
"""

import genesis as gs
import torch

from genesis_leg.cables import rotate
from genesis_leg.leg_model import prepare_leg
from genesis_leg.leg_tendons import pad_routes, tendon_forces

INTEGRATORS = {"mjINT_EULER": "Euler", "mjINT_IMPLICITFAST": "implicitfast"}
# Physics steps per model timestep. At 2 ms the settled pose matches MuJoCo but motion rings about twice as long;
# at 0.5 ms it follows MuJoCo closely. See "Physics steps" in genesis_leg/README.md.
DEFAULT_SUBSTEPS = 4


def init_genesis(backend="auto", seed=0):
    """Start Genesis on "gpu", "cpu" or "auto" (GPU if PyTorch sees one). Does nothing if already started."""
    if gs._initialized:
        return
    if backend == "auto":
        backend = "gpu" if torch.cuda.is_available() else "cpu"
    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu, seed=seed, precision="32", logging_level="warning")


class GenesisLeg:
    def __init__(self, xml, num_envs=1, substeps=DEFAULT_SUBSTEPS, show_viewer=False):
        """substeps: physics steps per model timestep. More follows MuJoCo more closely and costs proportionally."""
        self.model_xml, self.routes = prepare_leg(xml)
        self.num_envs = num_envs
        self.substeps = substeps
        self.dt = self.routes["timestep"]
        self.show_viewer = show_viewer
        self.device = gs.device
        integrator = INTEGRATORS.get(self.routes["integrator"])
        if integrator is None:
            raise ValueError(f"Genesis has no equivalent of {self.routes['integrator']}")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt / substeps, substeps=1, gravity=tuple(self.routes["gravity"]),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                enable_self_collision=False,
                enable_mujoco_compatibility=True,
                integrator=getattr(gs.integrator, integrator),
            ),
            viewer_options=gs.options.ViewerOptions(camera_pos=(1.0, -1.0, 0.2), camera_lookat=(0, 0, -0.25)),
            show_viewer=show_viewer,
        )
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(self.model_xml), default_armature=None))
        self.scene.build(n_envs=num_envs)

        tendons = self.routes["tendons"]
        self.tendon_names = [tendon["name"] for tendon in tendons]
        # Genesis can reorder links during import, so look them up by name.
        links, mask = pad_routes([[self.robot.get_link(b).idx_local for b in t["bodies"]] for t in tendons])
        offsets, _ = pad_routes([t["offsets"] for t in tendons])
        self.node_links = links.to(self.device)
        self.mask = mask.to(self.device)
        self.offsets = offsets.to(self.device, torch.float32)
        for name in ("kp", "kv", "fmax", "ctrl_low", "ctrl_high"):
            setattr(self, name, self._tensor([tendon[name] for tendon in tendons]))
        self.tip_link = self.robot.get_link(self.routes["tip_body"]).idx_local
        self.tip_local = self._tensor(self.routes["tip_local"])
        self.ctrl = self.ctrl_high.expand(num_envs, -1).clone()
        self.tension = torch.zeros_like(self.ctrl)
        self.length = torch.zeros_like(self.ctrl)
        self.length_rate = torch.zeros_like(self.ctrl)
        self._update_tendons()

    def _tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def pull_to_ctrl(self, pull):
        """Pull fraction per tendon: 0 is the rest length (no force from the servo), 1 is the full stroke."""
        return self.ctrl_high - pull.clamp(0, 1) * (self.ctrl_high - self.ctrl_low)

    def _update_tendons(self):
        """Refresh tendon state and return the (force, torque) to apply to every link, about each link origin."""
        quat = self.robot.get_links_quat()[:, self.node_links]
        offset = rotate(quat, self.offsets[None].expand(self.num_envs, -1, -1, -1))
        points = self.robot.get_links_pos()[:, self.node_links] + offset
        velocity = self.robot.get_links_vel()[:, self.node_links] + torch.linalg.cross(
            self.robot.get_links_ang()[:, self.node_links], offset,
        )
        force, self.tension, self.length, self.length_rate = tendon_forces(
            points, velocity, self.mask, self.ctrl, self.kp, self.kv, self.fmax,
        )
        # Several tendons pass through each link. Sum their forces and moments.
        forces = torch.zeros(self.num_envs, self.robot.n_links, 3, device=self.device)
        torques = torch.zeros_like(forces)
        idx = self.node_links.flatten()[None, :, None].expand(self.num_envs, -1, 3)
        forces.scatter_add_(1, idx, force.flatten(1, 2))
        torques.scatter_add_(1, idx, torch.linalg.cross(offset, force).flatten(1, 2))
        return forces, torques

    def step(self, ctrl=None):
        """Advance one model timestep toward target tendon lengths `ctrl` (default: keep the previous target).

        Afterward, tension and length are from the start of the last physics step, like MuJoCo's actuator_force.
        """
        if ctrl is not None:
            self.ctrl = torch.minimum(torch.maximum(ctrl.to(self.device), self.ctrl_low), self.ctrl_high)
        for substep in range(self.substeps):
            forces, torques = self._update_tendons()
            self.robot.apply_links_external_wrench(force=forces, torque=torques)
            self.scene.step(update_visualizer=self.show_viewer and substep == self.substeps - 1)

    def tip(self):
        return self.robot.get_links_pos()[:, self.tip_link] + rotate(
            self.robot.get_links_quat()[:, self.tip_link],
            self.tip_local.expand(self.num_envs, -1),
        )
