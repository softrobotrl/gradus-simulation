"""Board-pushing task: the upward-facing tendon leg pushes a board resting on rails above it.

The leg can only move the board by friction at its tip. The board rests on rails a few mm below the top of the
straightened leg, so pushing needs a stepping gait: straighten to lift the board off the rails, sweep the tip in
the push direction, bend to drop the board back onto the rails, swing back underneath, repeat.

    from pipeline.board_env import make_board_env
    env = make_board_env("pipeline/configs/opencr_tdcr_spatial.yaml", num_envs=256)  # MJX on GPU, else C MuJoCo
    obs = env.reset()["policy"]
    obs, reward, done, extras = env.step(actions)     # actions: (num_envs, 3) in [-1, 1]

Two backends, same task code and interface: BoardPushEnv (MJX; fast on a GPU) and BoardPushEnvCPU (C MuJoCo
in CPU threads; much faster than MJX when there is no GPU).

Full guide (physics, observation table, reward terms, every setting, SAC usage): pipeline/BOARD_TASK.md
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from pipeline.cpu_env import CpuTendonLegEnv
from pipeline.env import JAXTendonLegEnv, State, TendonLegEnv, gpu_problem


@dataclass(frozen=True)
class BoardConfig:
    """Board, rails and tip pad. Values marked ASSUMED are guesses until the real rig is measured."""

    length: float = 0.40          # m, board size along x (the default push direction)
    width: float = 0.15           # m, along y
    thickness: float = 0.012      # m
    mass: float = 0.1             # kg, ASSUMED (thin plywood). 0.2 kg also pushes in testing, but lifts less
    rail_gap: float = 0.003       # m: how far the straight leg's pad reaches above the board's resting underside
    rail_friction: float = 0.2    # N, ASSUMED: friction holding the board on the rails (x/y slide frictionloss)
    rail_damping: float = 0.5     # N*s/m, ASSUMED: stops the board coasting after a push
    max_lift: float = 0.03        # m, guides stop the board rising more than this above the rails
    pad_radius: float = 0.004     # m, ASSUMED: rubber-tipped sphere on the top disk
    friction: float = 0.8         # ASSUMED: pad-on-board sliding friction coefficient (rubber on wood)
    contact_timeconst: float = 0.005  # s, contact stiffness (MuJoCo solref). Default 0.02 lets the pad sink 1.6 mm
    start_offset: float = 0.02    # m, board centre starts uniformly within +-this of the leg in x and y
    edge_margin: float = 0.03     # m, episode ends (truncated) once the leg is this close to a board edge


class JAXBoardPushEnv(JAXTendonLegEnv):
    """Pure-functional JAX board-pushing env (one env; jit/vmap it yourself, or use BoardPushEnv)."""

    def __init__(self, config_path: str | Path, board: BoardConfig = BoardConfig(), random_direction: bool = False,
                 progress_reward: float = 100.0, lateral_cost: float = 50.0, rate_cost: float = 0.05,
                 tension_cost: float = 0.0, episode_s: float = 8.0, **kwargs):
        """
        board: physical setup, see BoardConfig.
        random_direction: False pushes along +x every episode; True draws a new direction per episode (the
            policy sees it in the observation).
        progress_reward: reward per m/s of board speed along the push direction (100 = 1 per cm/s).
        lateral_cost: penalty per m/s of board speed across the push direction.
        rate_cost: penalty on servo command changes, as a fraction of the max change per step (smoothness).
        tension_cost: penalty on mean squared tendon tension as a fraction of the servo's max pull (0 = off). The
            servos can pull ~100 N; a policy left unpenalized will use all of it, which real cables may not survive.
        kwargs: passed to JAXTendonLegEnv (effort_cost, init_servo_angle, init_bend_deg, device, ...).
        """
        # Set before super().__init__: it calls _load_model and traces reset, which use these.
        self.board = board
        self.random_direction = random_direction
        self.progress_reward = progress_reward
        self.lateral_cost = lateral_cost
        self.rate_cost = rate_cost
        self.tension_cost = tension_cost
        super().__init__(config_path, episode_s=episode_s, **kwargs)

    # ----- scene -----

    def _load_model(self, xml_path: Path) -> mujoco.MjModel:
        """robot.xml + a contact pad on the top disk + a board on 3 slide joints whose z lower limit is the rail."""
        b = self.board
        spec = mujoco.MjSpec.from_file(str(xml_path))
        robot = spec.compile()
        data = mujoco.MjData(robot)
        mujoco.mj_forward(robot, data)
        tip = robot.site("tip").id
        top_disk = spec.body(robot.body(robot.site_bodyid[tip]).name)
        # Straight-leg pad top minus the gap = where the board's underside sits when it rests on the rails.
        self.rail_z = float(data.site_xpos[tip, 2]) + b.pad_radius - b.rail_gap

        # Only the pad and the board collide (pad contype 1 meets board conaffinity 1); every robot geom is 0/0.
        contact = dict(condim=3, friction=[b.friction, 0.005, 0.0001], solref=[b.contact_timeconst, 1])
        top_disk.add_geom(name="pad", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[b.pad_radius, 0, 0],
                          pos=robot.site_pos[tip], contype=1, conaffinity=0, mass=0, rgba=[0.1, 0.1, 0.1, 1],
                          **contact)

        board = spec.worldbody.add_body(name="board", pos=[0, 0, self.rail_z + b.thickness / 2])
        for name, axis in (("board_x", [1, 0, 0]), ("board_y", [0, 1, 0])):
            board.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis,
                            frictionloss=b.rail_friction, damping=b.rail_damping)
        # z = 0 is "resting on the rails": the joint's lower limit plays the rails, so no rail geoms are needed.
        # solref_limit: rails as stiff as the pad contact (MuJoCo's default limit lets the board sink 0.4 mm).
        board.add_joint(name="board_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[0, b.max_lift],
                        limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=b.rail_damping,
                        solref_limit=[b.contact_timeconst, 1])
        board.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, mass=b.mass,
                       size=[b.length / 2, b.width / 2, b.thickness / 2], contype=0, conaffinity=1,
                       rgba=[0.75, 0.55, 0.3, 0.7], **contact)

        self.spec = spec
        model = spec.compile()
        board_joints = [model.joint(n).id for n in ("board_x", "board_y", "board_z")]
        self.board_qpos = jp.array([model.jnt_qposadr[j] for j in board_joints])
        self.board_dof = jp.array([model.jnt_dofadr[j] for j in board_joints])
        self.robot_qpos = jp.array(np.setdiff1d(np.arange(model.nq), np.asarray(self.board_qpos)))
        self.robot_dof = jp.array(np.setdiff1d(np.arange(model.nv), np.asarray(self.board_dof)))
        self.pad = model.geom("pad").id
        self.max_force = float(np.abs(model.actuator_forcerange).max())  # servo torque limit (N*m at the drum)
        return model

    def write_scene_xml(self, path: str | Path | None = None) -> Path:
        """Save the full scene (robot + pad + board) as XML for `python -m mujoco.viewer --mjcf=...`."""
        path = Path(path) if path else Path(self.spec.modelfiledir) / "board_scene.xml"
        path.write_text(self.spec.to_xml())
        return path

    def _reset_scene(self, key: jax.Array, qpos: jax.Array) -> jax.Array:
        """Board at a random xy offset over the leg, resting on the pad top (it settles in the first step)."""
        xy = jax.random.uniform(key, (2,), minval=-self.board.start_offset, maxval=self.board.start_offset)
        return qpos.at[self.board_qpos].set(jp.array([xy[0], xy[1], self.board.rail_gap]))

    # ----- task -----

    def _init_info(self, key: jax.Array, data: mjx.Data) -> dict:
        if self.random_direction:
            angle = jax.random.uniform(key, (), maxval=2 * jp.pi)
            direction = jp.array([jp.cos(angle), jp.sin(angle)])
        else:
            direction = jp.array([1.0, 0.0])
        zero = jp.zeros(())
        return {"terminated": zero, "direction": direction, "board_xy": data.qpos[self.board_qpos[:2]],
                "metrics": {"board_speed": zero, "lateral_speed": zero, "lifted": zero, "off_board": zero}}

    def _task(self, state: State, data: mjx.Data, target: jax.Array):
        direction = state.info["direction"]
        board_xy = data.qpos[self.board_qpos[:2]]
        velocity = (board_xy - state.info["board_xy"]) / self.dt  # m/s over this control step
        along = velocity @ direction
        across = jp.abs(velocity[0] * direction[1] - velocity[1] * direction[0])
        rate = jp.mean(jp.square((target - state.target) / self.max_step))  # 1 = every servo at full speed

        tension = jp.mean(jp.square(data.actuator_force / self.max_force))  # 1 = every servo at full pull

        reward = (self.progress_reward * along - self.lateral_cost * across
                  - self.effort_cost * self._effort(target) - self.rate_cost * rate - self.tension_cost * tension)

        reach = jp.array([self.board.length, self.board.width]) / 2 - self.board.edge_margin
        # Ran out of board (the leg sits at x = y = 0). A truncation, not a failure: pushing far is the goal, and the
        # policy can't see the board edges, so treating it as "future worth nothing" would just confuse the critic.
        off_board = jp.any(jp.abs(board_xy) > reach)
        lifted = data.qpos[self.board_qpos[2]] > 0.0005
        info = {"terminated": jp.zeros(()), "direction": direction, "board_xy": board_xy,
                "metrics": {"board_speed": along, "lateral_speed": across, "lifted": lifted.astype(jp.float32),
                            "off_board": off_board.astype(jp.float32)}}
        return reward, jp.zeros(()), off_board, info

    def _obs_parts(self, data: mjx.Data, target: jax.Array, info: dict) -> list[tuple[str, jax.Array]]:
        """Named observation pieces, in order. obs_layout lists their index ranges."""
        lift = data.qpos[self.board_qpos[2]]
        pad_top = data.geom_xpos[self.pad, 2] + self.board.pad_radius
        return [
            ("robot_qpos", data.qpos[self.robot_qpos]),       # ball-joint quaternions (4 each) + 3 cable slides (m)
            ("robot_qvel", data.qvel[self.robot_dof]),         # ball-joint angular velocities (rad/s) + cables (m/s)
            ("servo_target", target),                          # rate-limited servo angle setpoints (rad)
            ("tendon_tension", data.actuator_force / self.cfg.servo.drum_radius),  # N, what motor current shows
            ("tip_pos", data.site_xpos[self.tip]),             # m, world frame; the leg base is at the origin
            ("board_vel", data.qvel[self.board_dof]),          # m/s, x y z
            ("board_lift", lift[None]),                        # m above the rails (0 = resting on them)
            ("pad_gap", (self.rail_z + lift - pad_top)[None]),  # m from pad top to board underside; <= 0 = touching
            ("direction", info["direction"]),                  # unit vector of the push direction (x, y)
        ]

    def _obs(self, data: mjx.Data, target: jax.Array, info: dict) -> jax.Array:
        return jp.concatenate([v for _, v in self._obs_parts(data, target, info)])

    @property
    def obs_layout(self) -> dict[str, slice]:
        """Name -> slice into the observation vector, e.g. obs[..., env.obs_layout["board_vel"]]."""
        names = []

        def sizes(key):  # shape-only trace: no physics runs
            state = self.reset(key)
            parts = self._obs_parts(state.data, state.target, state.info)
            names[:] = [name for name, _ in parts]
            return [value for _, value in parts]

        shapes = jax.eval_shape(sizes, jax.random.PRNGKey(0))  # fills `names` while tracing
        layout, start = {}, 0
        for name, value in zip(names, shapes):
            layout[name] = slice(start, start + value.shape[0])
            start += value.shape[0]
        return layout


class BoardPushEnv(TendonLegEnv):
    """Batched PyTorch board-pushing env (rsl-rl VecEnv interface). kwargs go to JAXBoardPushEnv."""

    jax_env_cls = JAXBoardPushEnv

    @property
    def obs_layout(self) -> dict[str, slice]:
        return self.jax_env.obs_layout


class BoardPushEnvCPU(CpuTendonLegEnv):
    """Batched PyTorch board-pushing env with C MuJoCo physics in CPU threads. kwargs go to JAXBoardPushEnv."""

    jax_env_cls = JAXBoardPushEnv

    @property
    def obs_layout(self) -> dict[str, slice]:
        return self.jax_env.obs_layout


def make_board_env(config_path: str | Path, num_envs: int, backend: str = "auto", **kwargs):
    """The fastest board env for this machine.

    backend "auto": MJX on the GPU when JAX and PyTorch can both use one, otherwise C MuJoCo on CPU threads.
    "mjx": BoardPushEnv (kwargs may include device="auto"|"gpu"|"cpu"). "cpu": BoardPushEnvCPU.
    """
    if backend not in ("auto", "mjx", "cpu"):
        raise ValueError(f"backend must be 'auto', 'mjx' or 'cpu', got {backend!r}")
    if backend == "auto":
        problem = gpu_problem()
        backend = "cpu" if problem else "mjx"
        if problem:
            print(f"make_board_env: {problem}; using C MuJoCo on CPU threads")
    if backend == "mjx":
        return BoardPushEnv(config_path, num_envs, **kwargs)
    env = BoardPushEnvCPU(config_path, num_envs, **kwargs)
    print(f"BoardPushEnvCPU: running on CPU, {env.num_threads} physics threads")
    return env
