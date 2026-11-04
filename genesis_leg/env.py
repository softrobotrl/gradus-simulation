"""Train a tendon-driven leg to push a board in Genesis.

Genesis handles the joints and contact. Cable forces are calculated in PyTorch
and applied at each physics step.
"""

import math
from dataclasses import asdict, dataclass

import genesis as gs
import torch
from tensordict import TensorDict

from genesis_leg.cables import cable_forces, rotate
from genesis_leg.model import DEFAULT_CONFIG, build_scene


@dataclass
class LegConfig:
    robot_config: str = DEFAULT_CONFIG
    num_envs: int = 256
    episode_s: float = 8.0
    cable_stiffness: float = 2000.0  # N/m
    cable_damping: float = 1.0      # N*s/m
    action_filter: float = 0.2
    progress_reward: float = 100.0
    lateral_cost: float = 50.0
    effort_cost: float = 0.01
    rate_cost: float = 0.05

    def __post_init__(self):
        values = (
            self.episode_s, self.cable_stiffness, self.cable_damping,
            self.action_filter, self.progress_reward, self.lateral_cost,
            self.effort_cost, self.rate_cost,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Environment settings must be finite")
        if (
            self.num_envs < 1
            or self.episode_s <= 0
            or self.cable_stiffness <= 0
            or self.cable_damping < 0
        ):
            raise ValueError("Require positive num_envs, episode_s, stiffness; damping >= 0")
        if not 0 < self.action_filter <= 1:
            raise ValueError("action_filter must be in (0, 1]")


class GenesisLegEnv:
    def __init__(self, cfg: LegConfig, show_viewer=False):
        self.cfg = asdict(cfg)
        self.settings = cfg
        self.num_envs = cfg.num_envs
        self.device = gs.device
        self.show_viewer = show_viewer
        xml, self.metadata, self.robot_cfg = build_scene(cfg.robot_config)
        sim = self.robot_cfg.sim
        self.dt = sim.control_dt
        self.substeps = round(self.dt / sim.timestep)
        if self.substeps < 1 or not math.isclose(self.substeps * sim.timestep, self.dt):
            raise ValueError("control_dt must be an integer multiple of timestep")
        self.max_episode_length = math.ceil(cfg.episode_s / self.dt)
        self.num_actions = len(self.metadata["cables"])
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim.timestep, substeps=1),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True,
                enable_adjacent_collision=False,
                enable_mujoco_compatibility=True,
                constraint_timeconst=0.005,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.45, -0.55, 0.35),
                camera_lookat=(0, 0, 0.08),
            ),
            show_viewer=show_viewer,
        )
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(xml), default_armature=None))
        self.scene.build(n_envs=self.num_envs)

        # Genesis can reorder joints during import, so look them up by name.
        cable_joints = [self.robot.get_joint(f"cable{k}") for k in range(self.num_actions)]
        board_joints = [self.robot.get_joint(f"board_{axis}") for axis in "xyz"]
        self.cable_dofs = [joint.dofs_idx_local[0] for joint in cable_joints]
        self.cable_qs = [joint.qs_idx_local[0] for joint in cable_joints]
        self.board_qs = [joint.qs_idx_local[0] for joint in board_joints]
        self.board_dofs = [joint.dofs_idx_local[0] for joint in board_joints]
        self.ball_qs = [
            q
            for i in range(1, self.metadata["n_disks"])
            for q in self.robot.get_joint(f"joint{i}").qs_idx_local
        ]
        cables = self.metadata["cables"]
        self.node_links = self._tensor(
            [[self.robot.get_link(name).idx_local for name in cable["bodies"]] for cable in cables],
            dtype=torch.long,
        )
        self.offsets = self._tensor([cable["offsets"] for cable in cables])
        self.rest_lengths = self._tensor([cable["rest_length"] for cable in cables])
        self.tip_link = self.robot.get_link(self.metadata["tip_body"]).idx_local
        self.tip_local = self._tensor(self.metadata["tip_local"])
        self.initial_qpos = self.robot.get_qpos().clone()
        self.targets = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.tensions = torch.zeros_like(self.targets)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obs_buf = None
        self.reset()
        self.num_obs = self.obs_buf.shape[-1]

    def _tensor(self, values, dtype=torch.float32):
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def _geometry(self):
        pos = self.robot.get_links_pos()
        quat = self.robot.get_links_quat()
        offset = rotate(quat[:, self.node_links], self.offsets[None].expand(self.num_envs, -1, -1, -1))
        points = pos[:, self.node_links] + offset
        omega = self.robot.get_links_ang()[:, self.node_links]
        velocities = self.robot.get_links_vel()[:, self.node_links] + torch.linalg.cross(omega, offset)
        return pos, offset, points, velocities

    def _apply_cables(self):
        _, offset, points, velocity = self._geometry()
        force, self.tensions = cable_forces(
            points,
            velocity,
            self.rest_lengths,
            self.settings.cable_stiffness,
            self.settings.cable_damping,
        )
        # Several guides can belong to one link. Sum their forces and moments
        # about the link origin; Genesis accounts for the offset to its COM.
        shape = (self.num_envs, self.robot.n_links, 3)
        forces = torch.zeros(shape, device=self.device)
        torques = torch.zeros_like(forces)
        idx = self.node_links.flatten()[None, :, None].expand(self.num_envs, -1, 3)
        forces.scatter_add_(1, idx, force.flatten(1, 2))
        torques.scatter_add_(1, idx, torch.linalg.cross(offset, force).flatten(1, 2))
        self.robot.apply_links_external_wrench(force=forces, torque=torques)

    def _observe(self):
        q = self.robot.get_qpos()
        v = self.robot.get_dofs_velocity()
        tip = self.robot.get_links_pos()[:, self.tip_link] + rotate(
            self.robot.get_links_quat()[:, self.tip_link],
            self.tip_local.expand(self.num_envs, -1),
        )
        max_pull = self.robot_cfg.servo.max_torque / self.robot_cfg.servo.drum_radius
        # Include board position so the policy can see when it approaches an edge.
        self.obs_buf = torch.cat((q, v * 0.05, self.targets, self.tensions / max_pull, tip), dim=-1)
        return q, v

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset(self, envs_idx=None):
        idx = torch.arange(self.num_envs, device=self.device) if envs_idx is None else envs_idx
        if len(idx) == 0:
            return self.get_observations()
        q = self.initial_qpos[idx].clone()
        q[:, self.board_qs[:2]] = (torch.rand(len(idx), 2, device=self.device) * 2 - 1) * 0.02
        q[:, self.board_qs[2]] = 0.003
        self.robot.set_qpos(q, envs_idx=idx, zero_velocity=True)
        self.targets[idx] = 0
        self.tensions[idx] = 0
        self.episode_length_buf[idx] = 0
        self.episode_return[idx] = 0
        self._observe()
        return self.get_observations()

    def _advance_physics(self):
        servo = self.robot_cfg.servo
        self.peak_tension = torch.zeros(self.num_envs, device=self.device)
        for substep in range(self.substeps):
            angle = self.robot.get_qpos()[:, self.cable_qs] / servo.drum_radius
            speed = self.robot.get_dofs_velocity(self.cable_dofs) / servo.drum_radius
            torque = servo.kp * (self.targets - angle) - servo.kv * speed
            torque = torque.clamp(-servo.max_torque, servo.max_torque)
            self.robot.control_dofs_force(torque / servo.drum_radius, self.cable_dofs)
            self._apply_cables()
            self.peak_tension = torch.maximum(self.peak_tension, self.tensions.max(dim=-1).values)
            self.scene.step(
                update_visualizer=self.show_viewer and substep == self.substeps - 1,
            )

    def step(self, actions):
        actions = actions.to(self.device)
        if actions.shape != self.targets.shape or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {tuple(self.targets.shape)}")

        actions = actions.clamp(-1, 1)
        servo = self.robot_cfg.servo
        low, high = servo.angle_range
        desired = torch.where(actions >= 0, actions * high, -actions * low)
        previous = self.targets.clone()
        max_step = servo.max_speed * self.dt
        target_change = (desired - self.targets) * self.settings.action_filter
        self.targets += target_change.clamp(-max_step, max_step)

        before = self.robot.get_qpos()[:, self.board_qs[:2]].clone()
        self._advance_physics()
        q, _ = self._observe()
        velocity = (q[:, self.board_qs[:2]] - before) / self.dt
        cfg = self.settings
        effort = (self.targets / max(abs(a) for a in servo.angle_range)).square().mean(-1)
        action_rate = ((self.targets - previous) / max_step).square().mean(-1)
        reward = (
            cfg.progress_reward * velocity[:, 0]
            - cfg.lateral_cost * velocity[:, 1].abs()
            - cfg.effort_cost * effort
            - cfg.rate_cost * action_rate
        )
        self.episode_length_buf += 1
        balls = q[:, self.ball_qs].reshape(self.num_envs, -1, 4)
        bend = 2 * torch.acos(balls[..., 0].abs().clamp(0, 1))
        finite = torch.isfinite(self.obs_buf).all(-1) & torch.isfinite(reward)
        failed = ~finite | (bend > math.radians(self.robot_cfg.chain.joint.range_deg)).any(-1)
        failed |= self.scene.rigid_solver.get_error_envs_mask()
        off_board = (q[:, self.board_qs[:2]].abs() > self._tensor([0.17, 0.045])).any(-1)
        timeout = (self.episode_length_buf >= self.max_episode_length) | off_board
        done = failed | timeout
        self.reset_buf = done.clone()
        # Keep the last observation before reset for value bootstrapping.
        final_obs = torch.nan_to_num(self.obs_buf.clone())
        reward = torch.where(finite, reward, torch.full_like(reward, -10.0))
        self.episode_return += reward
        extras = {
            "time_outs": (timeout & ~failed).float(),
            "final_observation": final_obs,
            "peak_tension": self.peak_tension,
            "log": {
                "/board_speed": velocity[:, 0].mean(),
                "/tension": self.tensions.mean(),
                "/max_tension": self.peak_tension.max(),
                "/failure_rate": failed.float().mean(),
            },
        }
        if done.any():
            extras["episode"] = {
                "return": torch.nan_to_num(self.episode_return[done]).mean(),
                "failure_rate": failed[done].float().mean(),
                **extras["log"],
            }
            self.reset(done.nonzero(as_tuple=False).flatten())
        return self.get_observations(), reward, done, extras
