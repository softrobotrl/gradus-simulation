"""CPU MuJoCo env: demo leg on a one-dimensional sliding rig.

The hip is bolted to a carriage with a single prismatic joint along +X.  The
only ground contact is a small sphere at the foot, so the leg has to bend
itself with the eight tendon servos to push the carriage back and forth.

Two observation groups (see ``single_leg_demo/STATES.md``):

* ``policy`` (actor, runs on hardware) with ``obs_mode="hardware"``: only what
  the real rig can know, last ``obs_history`` steps stacked oldest first:
  target velocity (1), phase sin/cos (2), previous action (8), last servo
  targets sent (8), servo encoder positions (8), servo loads (8, only with
  ``servo_load_obs``), rail IMU accelerometer (3) and gyro (3): 33 values per
  control step, 41 with loads.  The confirmed hardware (2026-11-25) uses PWM
  servos, which report no load, and AS5600 encoders (``encoder_counts_per_rev``).
* ``critic`` (training only): the full 115-value simulator state, i.e.
  slider position/velocity, target velocity, phase, 13 ball-joint rotation
  vectors and velocities, tendon lengths, tendon forces, servo targets and
  the previous action, plus the carriage speed the reward scores (1) and 13
  domain-randomized values (``dr_params``; ``critic_dr_params=False`` drops them).

``obs_mode="full"`` gives the actor the 115-value simulator state (legacy runs).

Actions are eight residuals on a fixed tendon pull ``pull_bias`` (each tendon
just taut with the leg straight, a one-off calibration on hardware):
``pull = pull_bias + residual_scale * action`` in ``[0, 1]``.

Servo model: ``servo_cmd`` is the target our controller sends (smoothed
50/50 each step).  The simulated servo's setpoint ``d.ctrl`` moves toward it
at no more than ``servo_speed`` (spool m/s), like a real servo's top speed.
"""

from __future__ import annotations

import copy
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from single_leg_demo import rewards

N_BALL = 13
N_TENDON = 8
FOOT_RADIUS = rewards.FOOT_RADIUS
HW_FRAME = 1 + 2 + 4 * N_TENDON + 3 + 3   # 41 values per hardware step with servo loads


def hw_frame_size(servo_load_obs: bool, n_servo: int = N_TENDON, imu_obs: bool = True) -> int:
    """Values per hardware observation step (see STATES.md): command and
    phase (3), then per servo: previous action, target sent, encoder position
    (and load), then the rail IMU (6, only with imu_obs)."""
    return 3 + n_servo * (4 if servo_load_obs else 3) + (6 if imu_obs else 0)


SERVO_LAYOUTS = ("independent", "paired_x", "single_x")


def servo_groups(m, layout: str) -> np.ndarray:
    """Servo index of each tendon actuator.

    ``independent``: one servo per tendon (8, runs before 2026-11-26).
    ``paired_x``: the confirmed leg, 4 double-spool servos.  Each spool pulls
    the two cables of one ring on one side of the travel direction (+X or -X).
    The holes sit at 45 degrees to the rail, so the two cables' sideways pulls
    cancel: the leg bends forward and back only.  Servo order: inner +X,
    inner -X, outer +X, outer -X.
    ``single_x``: cable holes aligned with the rail (use with leg_yaw_deg=45
    on this CAD).  4 servos, each pulling the one cable straight ahead of
    (+X) or behind (-X) the backbone on one ring; the sideways cables get no
    servo (-1 here; the env gives them zero force, as if absent).  Same servo
    order as paired_x.
    """
    if layout == "independent":
        return np.arange(m.nu)
    if layout not in SERVO_LAYOUTS:
        raise ValueError(f"servo_layout must be one of {SERVO_LAYOUTS}, got {layout!r}")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # hole positions at the hip, relative to the centre of their ring
    hole = {}
    for a in range(m.nu):
        t = int(m.actuator_trnid[a, 0])
        hole[a] = d.site_xpos[m.wrap_objid[m.tendon_adr[t]]][:2].copy()
    ring_of = {a: 0 if "inner" in m.tendon(int(m.actuator_trnid[a, 0])).name else 1 for a in range(m.nu)}
    centre = {r: np.mean([hole[a] for a in hole if ring_of[a] == r], axis=0) for r in (0, 1)}
    groups = np.zeros(m.nu, dtype=int)
    for a in range(m.nu):
        ring = ring_of[a]
        x, y = hole[a] - centre[ring]
        side = 2 * ring + (0 if x > 0 else 1)
        if layout == "paired_x":
            groups[a] = side
        else:
            groups[a] = side if abs(x) > 2 * abs(y) else -1     # on the rail axis, or sideways
    want = [2, 2, 2, 2] if layout == "paired_x" else [1, 1, 1, 1]
    if sorted(np.bincount(groups[groups >= 0], minlength=4)) != want:
        raise RuntimeError(f"{layout} expects {want[0]} cable(s) per servo, got {groups} "
                           "(for single_x, are the holes aligned with the rail? see leg_yaw_deg)")
    return groups
GRAVITY = 9.81
CMD_BUF = 3          # sent commands kept for the control delay (max 2 steps = 40 ms)


@dataclass
class RigConfig:
    model_path: str = "urdf/gradus/demo/rig.xml"
    num_envs: int = 64
    n_substeps: int = 10
    control_dt: float = 0.02
    episode_steps: int = 400
    rail_half: float = 0.45

    # Desired speed: the command alternates sign every cmd_period seconds.
    cmd_min: float = 0.03
    cmd_max: float = 0.12
    cmd_period: float = 2.0
    cmd_period_jitter: float = 0.20
    cmd_resample: bool = True        # new random magnitude at every reversal
    obs_cmd_scale: float = 0.16      # fixed normaliser, independent of cmd_max

    # Observations (see STATES.md).
    obs_mode: str = "hardware"       # "hardware": actor sees sensors only; "full": legacy 115-d
    # Confirmed hardware (2026-11-25): PWM servos report no load, so the actor
    # gets no load reading.  True = the 41-value frame of earlier runs.
    servo_load_obs: bool = False
    # Confirmed hardware: 4 double-spool servos per leg (servo_groups).
    # "independent" = one servo per tendon (8 actions, runs before 2026-11-26).
    servo_layout: str = "paired_x"
    # Servo torque (N*m) -> pull limit per cable = torque / spool_radius /
    # cables per servo.  ANNIMOS 80 kg (rated stall 80 kg*cm, about 7.8 N*m);
    # 4.0 assumes about half the rating is usable, and gives 200 N per cable,
    # the value in rig.xml.  0 = keep rig.xml's limit.
    servo_torque: float = 4.0
    # Turn the leg about the vertical axis relative to the rail (degrees).
    # 0 = as in the CAD: cable holes at 45 degrees to the rail (paired_x).
    # 45 = holes straight ahead, behind and to the sides (single_x).
    leg_yaw_deg: float = 0.0
    # AS5600 magnetic encoder on the spool: 4096 counts per revolution.  The
    # position reading is rounded to whole counts (always, not only with DR).
    # 0 = no rounding (runs before 2026-11-26).
    encoder_counts_per_rev: int = 4096
    spool_radius: float = 0.0135      # m; cable pulled per radian of spool turn (13.5 mm, 2026-11-26)
    # The confirmed rig has no IMU (2026-11-26): the actor gets no rail IMU.
    # True = the 6 IMU values of earlier runs.
    imu_obs: bool = False
    obs_history: int = 8             # hardware steps stacked for the actor
    gyro_scale: float = 4.0          # rad/s normaliser for the rail IMU gyro (clipped at ±2)
    acc_clip: float = 3.0            # g, rail IMU accelerometer clip

    # Reward.  The terms and presets live in rewards.py (start there, and see
    # REWARD.md); these fields are their weights and parameters.  Defaults are
    # reward v2; rewards.apply_preset sets v1/v3/v4.
    reward_preset: str = ""           # which preset set the fields below (provenance only)
    vel_weight: float = 2.0
    vel_sigma: float = 0.06
    vel_l2_weight: float = 0.02      # keeps a gradient during fast reversals
    progress_weight: float = 0.5
    rail_weight: float = 0.5
    foot_weight: float = 0.3
    lat_weight: float = 0.5
    effort_weight: float = 0.02
    tension_weight: float = 0.001
    action_l2_weight: float = 0.005   # keep raw actions out of hard saturation
    failure_penalty: float = 5.0

    # Reward v2: smoothness and hardware-damage costs.  Every weight below is
    # multiplied by the curriculum factor k (see _curriculum).  Weights were
    # chosen so one clearly bad step costs about 1 (one step of tracking
    # reward) while smooth motion of the same size costs a few percent.
    # ``train.py --reward v1`` restores the original reward exactly.
    cmd_ramp: float = 0.3             # m/s^2 rate limit on the command; 0 = step command
    rate_weight: float = 0.5          # mean (a_t - a_{t-1})^2       (v1: 0.02, no k)
    accel_weight: float = 2.5         # mean (a_t - 2a_{t-1} + a_{t-2})^2
    shock_weight: float = 0.8         # mean (dT per step / (0.1 Tmax))^2
    tension_limit_weight: float = 1.0
    tension_soft: float = 0.7         # fraction of Tmax where the tension cost starts
    bend_limit_weight: float = 1.0
    bend_soft: float = 0.8            # fraction of bend_limit where the bend cost starts
    impact_weight: float = 0.5
    foot_static_force: float = 31.0   # N, foot load holding the reset pose (measured)
    slip_weight: float = 0.2
    slip_scale: float = 0.05          # m/s of foot sliding that costs 1
    slip_cap: float = 5.0             # slip cost per step is capped at this many units
    # k = k0^(kd^iteration), the cost curriculum of Hwangbo et al. 2019 (they
    # used k0 = 0.3).  Here the costs also see PPO's exploration noise: at the
    # initial action std of 0.5 the v2 costs are ~4.8 per step at k = 1,
    # against ~1.1 of tracking reward.  k0 = 0.05 keeps that early cost ~0.24.
    # k = 0.30 at iteration 300, 0.51 at 500, 0.86 at 1000.
    curriculum_k0: float = 0.05
    curriculum_kd: float = 0.997
    curriculum_iter_steps: int = 24   # env steps per PPO iteration (--steps-per-env)
    # The costs above may take a step's reward down to 0 but not below, so
    # surviving is never worth less than failing.  Same idea as legged_gym's
    # only_positive_rewards, but the original (v1) terms are not clipped.
    bound_costs: bool = True
    # Reward v3: free bands before the foot-height and sideways penalties start.
    # With the hip 2 mm below the straight-leg contact height the foot touches
    # the floor anywhere within ~4 cm of under the hip, so the swing back to the
    # front has to go around the side (a lateral bend lifts the foot).  v1/v2
    # penalised exactly that path.  0 = the v1/v2 penalties unchanged.
    lat_free: float = 0.0             # m of sideways foot offset that costs nothing
    foot_free: float = 0.0            # m of foot lift that costs nothing
    # Reward v4: track the carriage speed averaged over about one stride
    # (exponential average, time constant in s) instead of the instantaneous
    # speed.  One leg can only move the carriage in pulses (push, glide,
    # swing), so the instantaneous speed swings 0 -> 0.3 m/s even when the
    # average is right, and the squared velocity term then scores any gait
    # below standing still.  0 = instantaneous speed (v1-v3).
    vel_avg_tau: float = 0.0
    # Reward v5: idling cost.  The 1 s average above lets a policy push hard
    # for a moment and then stand still.  `idle` charges the shortfall of the
    # speed averaged over a shorter window (idle_tau s) from the command, so
    # long pauses cost even when the 1 s average is close.  0 = off (v1-v4).
    idle_weight: float = 0.0
    idle_tau: float = 0.5
    # Rail (carriage) dry friction in N; < 0 keeps the value in rig.xml (0.2 N).
    rail_friction: float = -1.0

    # Termination / reset.
    bend_limit: float = 0.90          # rad, per ball joint
    foot_lift_limit: float = 0.12     # m, foot centre above the floor
    reset_bend: float = 0.012         # rad per ball joint, sign randomised
    reset_noise: float = 0.25         # scales reset perturbations
    reset_settle: int = 20            # physics steps after the reset pose

    residual_scale: float = 0.6

    # Servo model.  Assumed generic hobby servo (0.15 s / 60 deg, 10 mm spool)
    # until the real servo's datasheet is known; <= 0 disables the limit.
    servo_speed: float = 0.07         # m/s of tendon at the spool
    fixed_pull_bias: bool = True      # False: legacy per-reset bias from sim tendon lengths

    # Domain randomization, training only (see domain_randomization.md).
    # Drawn per episode; "×" ranges multiply the model's value.  dr_scale pulls
    # every draw toward nominal (0 = nominal, 1 = full range) for ramp-up.
    dr_enabled: bool = False
    dr_scale: float = 1.0
    dr_friction: tuple = (0.5, 1.3)          # foot-floor sliding friction (absolute)
    dr_rail_friction: tuple = (0.5, 2.0)     # × carriage frictionloss
    dr_rail_damping: tuple = (0.5, 2.0)      # × carriage damping
    dr_carriage_mass: tuple = (0.8, 1.2)     # × carriage mass
    # Rail (hip) height offset in m, for build/assembly error.  Lopsided on
    # purpose: above about +1 mm the straight leg no longer reaches the floor
    # (unworkable for any policy); down to -10 mm the leg still drives the rail.
    dr_rail_height: tuple = (-0.008, 0.0005)
    dr_segment_mass: tuple = (0.9, 1.1)      # × each leg body's mass
    dr_stiffness: tuple = (0.8, 1.2)         # × backbone (ball joint) stiffness
    dr_joint_damping: tuple = (0.7, 1.5)     # × ball joint damping
    dr_servo_kp: tuple = (0.7, 1.3)          # × servo position gain
    dr_servo_strength: tuple = (0.9, 1.1)    # × max force, per servo
    dr_servo_calib: float = 0.001            # ± m tendon zero error, per servo
    dr_battery: tuple = (0.8, 1.0)           # battery level (1 = full)
    dr_sag: tuple = (0.0, 0.25)              # extra voltage drop at full servo load
    dr_delay: tuple = (0, 10)                # control delay in 2 ms substeps (0-20 ms)
    dr_push_prob: float = 0.005              # chance per control step of a carriage push
    dr_push_force: float = 3.0               # ± N
    dr_push_steps: int = 5                   # push duration in control steps (0.1 s)
    # Sensor noise on the policy input (only when dr_enabled).
    dr_acc_noise: float = 0.01               # g, per step
    dr_acc_bias: float = 0.03                # ± g, per episode
    dr_gyro_noise: float = 0.01              # rad/s, per step
    dr_gyro_bias: float = 0.02               # ± rad/s, per episode
    dr_servo_counts: int = 2048              # unused since encoder_counts_per_rev (kept so old configs load)
    dr_pos_noise: float = 0.002              # servo position, fraction of stroke
    dr_load_noise: float = 0.03              # servo load, fraction of max force
    dr_load_scale: tuple = (0.8, 1.2)        # × load reading, per servo
    # Fix parameters by name, e.g. {"friction": 0.4, "battery": 0.8} (works
    # with dr_enabled False too: everything else stays nominal).
    dr_override: dict = field(default_factory=dict)
    # Critic (training only) also sees this episode's randomized values.
    critic_dr_params: bool = True
    # Critic also sees the carriage speed the reward scores (the ~1 s average under v4).
    critic_scored_speed: bool = True

    num_threads: int | None = None
    seed: int = 0


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternion (w, x, y, z) -> (..., 3) rotation vector."""
    w = np.abs(q[..., 0])
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, w[..., None])
    axis = v / np.maximum(n, 1e-9)
    return angle * axis


def _bend_angle(q: np.ndarray) -> np.ndarray:
    return 2.0 * np.arccos(np.clip(np.abs(q[..., 0]), 0.0, 1.0))


def _load_with_foot_sensors(path: str, leg_yaw_deg: float = 0.0) -> mujoco.MjModel:
    """Load the rig and add foot sensors for the reward (they do not change the physics).

    These are simulator-only (the reward runs in training only; the real leg has
    no foot sensor).  Read them by name, never by index: rig.xml has its own
    sensors (the rail IMU) ahead of these.

    foot_force   touch: floor normal force on the foot sphere (N)
    foot_linvel  linear velocity of the foot centre, world frame
    foot_angvel  angular velocity of the foot body, world frame
    """
    spec = mujoco.MjSpec.from_file(path)
    plain = mujoco.MjModel.from_xml_path(path)
    if leg_yaw_deg:
        # turn the whole leg (hip mount and everything below it, with its cable
        # holes) about the vertical axis, relative to the carriage and rail
        slider = plain.body("slider").id
        hip = [b for b in range(plain.nbody) if plain.body_parentid[b] == slider]
        if len(hip) != 1:
            raise RuntimeError(f"expected one leg body under the carriage, got {hip}")
        b = spec.body(plain.body(hip[0]).name)
        h = math.radians(leg_yaw_deg) / 2
        qz = np.array([math.cos(h), 0.0, 0.0, math.sin(h)])
        q0 = np.array(b.quat, dtype=float)
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = q0
        b.quat = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    foot = plain.geom("foot")
    body = spec.body(plain.body(int(foot.bodyid[0])).name)
    pos = [float(v) for v in foot.pos]
    radius = float(foot.size[0])
    body.add_site(name="foot_touch", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[radius * 1.2, 0, 0], pos=pos, group=5)
    body.add_site(name="foot_centre", pos=pos, size=[0.001, 0, 0], group=5)
    for name, kind, obj in (("foot_force", mujoco.mjtSensor.mjSENS_TOUCH, "foot_touch"),
                            ("foot_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL, "foot_centre"),
                            ("foot_angvel", mujoco.mjtSensor.mjSENS_FRAMEANGVEL, "foot_centre")):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=obj)
    return spec.compile()


class SingleLegRigEnv:
    def __init__(self, cfg: RigConfig):
        self.cfg = asdict(cfg)
        self.device = "cpu"
        self.num_envs = cfg.num_envs
        self.max_episode_length = cfg.episode_steps
        self.n_substeps = cfg.n_substeps
        self.dt = cfg.control_dt
        self.rng = np.random.default_rng(cfg.seed)

        m = _load_with_foot_sensors(cfg.model_path, cfg.leg_yaw_deg)
        if cfg.rail_friction >= 0.0:
            m.dof_frictionloss[m.jnt_dofadr[m.joint("slide").id]] = cfg.rail_friction
        self.m = m
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        if self.nu != N_TENDON:
            raise RuntimeError(f"expected {N_TENDON} tendon actuators, model has {self.nu}")
        # servo of each tendon actuator; the policy drives servos, not tendons
        # (-1 = a cable with no servo: single_x's sideways cables)
        self.servo_of = servo_groups(m, cfg.servo_layout)
        self.n_servo = int(self.servo_of.max()) + 1
        self.num_actions = self.n_servo
        unused = self.servo_of < 0
        # servo index per tendon for indexing per-servo arrays padded with one
        # extra entry (action 0, strength 1) for cables without a servo
        self.servo_idx = np.where(unused, self.n_servo, self.servo_of)
        counts = np.bincount(self.servo_of[~unused], minlength=self.n_servo)
        self.cables_per_servo = np.where(unused, 1, counts[self.servo_idx % max(self.n_servo, 1)])
        self.first_cable = np.array([np.flatnonzero(self.servo_of == s)[0] for s in range(self.n_servo)])
        # the cables of each servo's spool, (n_servo, cables per servo); None when 1 servo per tendon
        self.servo_cables = (np.array([np.flatnonzero(self.servo_of == s) for s in range(self.n_servo)])
                             if cfg.servo_layout != "independent" else None)
        if cfg.servo_torque > 0:
            m.actuator_forcerange[:, 0] = -cfg.servo_torque / cfg.spool_radius / self.cables_per_servo
        elif unused.any():
            raise ValueError("servo_layout single_x needs servo_torque > 0")

        self.l0 = m.actuator_ctrlrange[:, 1].copy()
        self.stroke = (m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0]).copy()
        self.fmax = (-m.actuator_forcerange[:, 0]).copy()   # unused cables keep a nominal scale
        m.actuator_forcerange[unused] = 0.0                  # cables without a servo pull nothing
        self.ctrl_lo = self.l0 - self.stroke

        self.slide_joint = m.joint("slide")
        self.slide_qadr = int(self.slide_joint.qposadr[0])
        self.ball_joints = [m.joint(f"{i:02d}") for i in range(N_BALL)]
        self.ball_qadr = np.array([int(j.qposadr[0]) for j in self.ball_joints])
        self.ball_dadr = np.array([int(j.dofadr[0]) for j in self.ball_joints])
        self.tip_id = m.site("tip").id
        self.qpos_straight = m.qpos0.copy()
        if cfg.obs_mode not in ("hardware", "full"):
            raise ValueError(f"obs_mode must be 'hardware' or 'full', got {cfg.obs_mode!r}")
        try:
            self.imu_acc_adr = int(m.sensor("imu_acc").adr[0])
            self.imu_gyro_adr = int(m.sensor("imu_gyro").adr[0])
        except KeyError as exc:
            raise RuntimeError(
                f"{cfg.model_path} has no rail IMU sensors; "
                "rebuild it with `python -m single_leg_demo.build_rig`") from exc

        # --- domain randomization ------------------------------------------ #
        # Physics parameters live on the MjModel, so randomized envs each get
        # their own copy.  Without randomization every env shares self.m.
        n = cfg.num_envs
        self.per_env_models = bool(cfg.dr_enabled or cfg.dr_override)
        # copies of m, so each keeps the reward's foot sensors and any rail_friction override
        self.models = ([copy.deepcopy(m) for _ in range(n)]
                       if self.per_env_models else [m] * n)
        self.nom = {k: getattr(m, k).copy() for k in (
            "geom_friction", "body_mass", "body_inertia", "body_pos", "dof_damping",
            "dof_frictionloss", "jnt_stiffness", "actuator_gainprm",
            "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange")}
        self.floor_geom = m.geom("floor").id
        self.foot_geom = m.geom("foot").id
        self.slider_body = m.body("slider").id
        self.leg_bodies = np.array([b for b in range(1, m.nbody) if b != self.slider_body])
        self.slide_dof = int(self.slide_joint.dofadr[0])
        self.ball_jnt = np.array([j.id for j in self.ball_joints])
        self.ball_dofs = np.concatenate([np.arange(a, a + 3) for a in self.ball_dadr])
        # per-episode draws (nominal values until _randomize runs)
        self.strength = np.ones((n, N_TENDON))
        self.calib = np.zeros((n, N_TENDON))      # m, physical = commanded + calib
        self.battery = np.ones(n)
        self.sag = np.zeros(n)
        self.delay = np.zeros(n, dtype=np.int32)  # substeps
        self.acc_bias = np.zeros((n, 3))
        self.gyro_bias = np.zeros((n, 3))
        self.load_scale = np.ones((n, N_TENDON))
        self.push_force = np.zeros(n)
        self.push_left = np.zeros(n, dtype=np.int32)
        self.volt = np.ones(n)                    # last supply voltage (fraction)

        self.datas = [mujoco.MjData(mi) for mi in self.models]
        self.t = np.zeros(cfg.num_envs, dtype=np.int32)
        self.cmd = np.zeros(cfg.num_envs, dtype=np.float64)
        self.cmd_mag = np.zeros(cfg.num_envs, dtype=np.float64)
        self.cmd_sign0 = np.ones(cfg.num_envs, dtype=np.float64)
        self.cmd_period = np.full(cfg.num_envs, cfg.cmd_period, dtype=np.float64)
        self.cmd_segment = np.zeros(cfg.num_envs, dtype=np.int32)
        # Fixed calibration: tendon pull with every tendon just taut and the
        # leg straight.  Hardware can reproduce this at startup.
        d_cal = mujoco.MjData(m)
        d_cal.qpos[:] = self.qpos_straight
        mujoco.mj_forward(m, d_cal)
        self.pull_bias_nominal = np.clip((self.l0 - d_cal.ten_length) / self.stroke, 0.0, 1.0)
        self.pull_bias = np.tile(self.pull_bias_nominal, (cfg.num_envs, 1))
        # What our controller sent (tendon-length units); d.ctrl is where the
        # simulated servo actually is heading, rate-limited toward this.
        self.servo_cmd = np.tile(self.l0 - self.stroke * self.pull_bias_nominal,
                                 (cfg.num_envs, 1))
        # sent commands: [this step, 1 step ago, 2 steps ago], for the control delay
        self.cmd_buf = np.repeat(self.servo_cmd[:, None, :], CMD_BUF, axis=1)
        self.prev_action = np.zeros((cfg.num_envs, self.n_servo), dtype=np.float64)
        self.prev_action2 = np.zeros((cfg.num_envs, self.n_servo), dtype=np.float64)

        # Reward v2 state.  cmd is the (ramped) command the policy sees and
        # tracks; cmd_raw is the square-wave target it ramps towards.
        self.cmd_raw = np.zeros(cfg.num_envs, dtype=np.float64)
        self.prev_tension = np.zeros((cfg.num_envs, N_TENDON), dtype=np.float64)
        self.vx_avg = np.zeros(cfg.num_envs, dtype=np.float64)
        self.vx_idle = np.zeros(cfg.num_envs, dtype=np.float64)
        self.peak_tension = np.zeros((cfg.num_envs, N_TENDON), dtype=np.float64)
        self.peak_foot_force = np.zeros(cfg.num_envs, dtype=np.float64)
        self.slip_sum = np.zeros(cfg.num_envs, dtype=np.float64)
        self.slip_n = np.zeros(cfg.num_envs, dtype=np.int32)
        self.floor_id = m.geom("floor").id
        self.foot_radius = float(m.geom("foot").size[0])
        self.foot_force_adr = int(m.sensor("foot_force").adr[0])
        self.foot_linvel_adr = int(m.sensor("foot_linvel").adr[0])
        self.foot_angvel_adr = int(m.sensor("foot_angvel").adr[0])
        self.env_steps = 0  # control steps taken by the whole batch, drives the curriculum
        self._k = self._curriculum()

        # encoder counts over one full stroke: counts/rev x stroke / (2 pi r)
        self.encoder_counts = (cfg.encoder_counts_per_rev * self.stroke / (2 * math.pi * cfg.spool_radius)
                               if cfg.encoder_counts_per_rev > 0 else None)
        # encoder counts over the stroke, per servo (both cables of a spool share it)
        self.servo_counts = (None if self.encoder_counts is None else
                             self.encoder_counts if self.servo_cables is None else
                             self.encoder_counts[self.servo_cables].mean(axis=1))
        self.hw_frame = hw_frame_size(cfg.servo_load_obs, self.n_servo, cfg.imu_obs)
        self.hist = np.zeros((cfg.num_envs, max(1, cfg.obs_history), self.hw_frame))
        self.num_critic_obs = int(self._obs_critic(0).shape[0])
        self.num_obs = (self.hw_frame * self.hist.shape[1] if cfg.obs_mode == "hardware"
                        else int(self._obs_full(0).shape[0]))
        self.obs = np.zeros((cfg.num_envs, self.num_obs), dtype=np.float32)
        self.critic_obs = np.zeros((cfg.num_envs, self.num_critic_obs), dtype=np.float32)

        threads = max(1, min(cfg.num_threads or os.cpu_count() or 1, cfg.num_envs))
        self.num_threads = threads
        self._chunks = [np.arange(i, cfg.num_envs, threads) for i in range(threads)]
        self._pool = ThreadPoolExecutor(threads)

        self.reset()

    # ------------------------------------------------------------------ #
    def _set_cmd(self, i: int) -> None:
        t = self.t[i] * self.dt
        period = max(self.cmd_period[i], 1e-3)
        segment = int(math.floor(t / period + 1e-9))
        if self.cfg["cmd_resample"] and segment != self.cmd_segment[i]:
            # A fixed per-episode amplitude lets the policy learn one open-loop
            # pump and ignore the command.  Resampling at every reversal keeps
            # the commanded magnitude informative.
            self.cmd_mag[i] = self.rng.uniform(self.cfg["cmd_min"], self.cfg["cmd_max"])
        self.cmd_segment[i] = segment
        sign = self.cmd_sign0[i] * (1.0 if segment % 2 == 0 else -1.0)
        self.cmd_raw[i] = sign * self.cmd_mag[i]
        ramp = self.cfg["cmd_ramp"]
        if ramp > 0.0:
            # Rate-limit the command so a reversal asks for an acceleration
            # the leg can deliver (the baseline reached ~0.17-0.31 m/s^2).
            # Called once per control step, and once at reset.
            step = ramp * self.dt
            self.cmd[i] += float(np.clip(self.cmd_raw[i] - self.cmd[i], -step, step))
        else:
            self.cmd[i] = self.cmd_raw[i]

    def _phase(self, i: int) -> float:
        period = max(self.cmd_period[i], 1e-3)
        return float((self.t[i] * self.dt / period) % 1.0)

    def _obs_noise(self, i: int) -> np.ndarray | None:
        """Env i's sensor noise for one observation step, in the order the
        policy input lists the sensors (acc 3, gyro 3, position, load), or
        None without randomization.  Drawn per env, in env order, so the
        random stream does not depend on how the frames are then computed."""
        if not self.cfg["dr_enabled"]:
            return None
        return self.rng.standard_normal(6 + 2 * self.n_servo)

    def _hw_frames(self, idx: np.ndarray, noise: list) -> np.ndarray:
        """One step of what the real rig can know, for envs ``idx`` (33 values
        per env, 41 with loads; STATES.md).

        Called after the physics step, so ``servo_cmd`` is the servo target
        that was sent for the last 20 ms, and ``prev_action`` is the policy
        output that produced it.
        """
        datas = [self.datas[i] for i in idx]
        ctrl = np.array([d.ctrl for d in datas])
        ten_length = np.array([d.ten_length for d in datas])
        force = np.array([d.actuator_force for d in datas])
        sens = np.array([d.sensordata for d in datas])
        target = np.clip((self.l0 - self.servo_cmd[idx]) / self.stroke, 0.0, 1.0)
        # Servo spool position: the servo heads for its rate-limited setpoint
        # (d.ctrl); a taut tendon holds the spool short of it, a slack tendon
        # lets it get there.
        setpoint = (self.l0 - ctrl) / self.stroke
        taut = (self.l0 - ten_length) / self.stroke
        # encoder reads in the robot's own (possibly mis-calibrated) zero
        position = np.minimum(taut, setpoint) + self.calib[idx] / self.stroke
        load = -force / self.fmax
        if self.servo_cables is not None:
            # one spool per servo: the tauter cable holds it back, the encoder
            # reads the spool; the servo carries all its cables' tension
            sc = self.servo_cables
            target = target[:, sc].mean(axis=2)
            position = position[:, sc].min(axis=2)
            load = load[:, sc].mean(axis=2)
        acc = sens[:, self.imu_acc_adr:self.imu_acc_adr + 3] / GRAVITY
        gyro = sens[:, self.imu_gyro_adr:self.imu_gyro_adr + 3].copy()
        if self.cfg["dr_enabled"]:                   # sensor noise, policy input only
            sc, S = self.cfg["dr_scale"], self.n_servo
            z = np.array(noise)
            acc = acc + self.acc_bias[idx] + self.cfg["dr_acc_noise"] * sc * z[:, 0:3]
            gyro = gyro + self.gyro_bias[idx] + self.cfg["dr_gyro_noise"] * sc * z[:, 3:6]
            position = position + self.cfg["dr_pos_noise"] * sc * z[:, 6:6 + S]
            load = (load * self.load_scale[idx][:, self.first_cable]
                    + self.cfg["dr_load_noise"] * sc * z[:, 6 + S:6 + 2 * S])
        if self.servo_counts is not None:            # the encoder reads whole counts
            position = np.round(position * self.servo_counts) / self.servo_counts
        # physical limits, same as the hardware controller must apply: a sensor
        # spike (impact, glitch) can never feed the policy an extreme value
        position = np.clip(position, 0.0, 1.0)
        load = np.clip(load, 0.0, 1.5)
        acc = np.clip(acc, -self.cfg["acc_clip"], self.cfg["acc_clip"])
        gyro = np.clip(gyro / self.cfg["gyro_scale"], -2.0, 2.0)   # ±2 × gyro_scale rad/s
        return np.concatenate([
            self._cmd_phase(idx),     # target velocity, phase sin/cos
            self.prev_action[idx],    # previous action (policy output)
            target,                   # last servo targets sent
            position,                 # servo encoder positions
            *([load] if self.cfg["servo_load_obs"] else []),   # servo loads (not on PWM servos)
            *([acc, gyro] if self.cfg["imu_obs"] else []),   # rail IMU (none on the confirmed rig)
        ], axis=1)

    def _cmd_phase(self, idx: np.ndarray) -> np.ndarray:
        """(len(idx), 3): target velocity (scaled), sin and cos of the phase."""
        out = np.empty((len(idx), 3))
        scale = max(self.cfg["obs_cmd_scale"], 1e-3)
        for j, i in enumerate(idx):
            phase = self._phase(i)
            out[j] = (self.cmd[i] / scale, math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase))
        return out

    def _push_obs(self, i: int, fresh: bool = False) -> None:
        """Advance env i's history and refresh its actor and critic observations."""
        self._push_obs_many(np.array([i]), [self._obs_noise(i)], fresh)

    def _push_obs_many(self, idx: np.ndarray, noise: list, fresh: bool = False) -> None:
        """Advance the history of envs ``idx`` and refresh their observations.
        ``noise``: one ``_obs_noise`` draw per env (None entries without DR)."""
        frame = self._hw_frames(idx, noise)
        if fresh:
            self.hist[idx] = frame[:, None, :]          # reset: fill history with this step
        else:
            self.hist[idx, :-1] = self.hist[idx, 1:]    # oldest first, newest last
            self.hist[idx, -1] = frame
        full = self._obs_full_many(idx)
        self.critic_obs[idx] = self._obs_critic_many(idx, full)
        self.obs[idx] = (self.hist[idx].reshape(len(idx), -1) if self.cfg["obs_mode"] == "hardware"
                         else full)

    def _obs_critic(self, i: int) -> np.ndarray:
        return self._obs_critic_many(np.array([i]), self._obs_full_many(np.array([i])))[0]

    def _obs_full(self, i: int) -> np.ndarray:
        return self._obs_full_many(np.array([i]))[0]

    def _obs_critic_many(self, idx: np.ndarray, full: np.ndarray) -> np.ndarray:
        """Critic input: full sim state (115) + scored speed (1, optional) + randomized values (13, optional).

        The scored speed is the carriage speed the reward's velocity terms use
        (the ~1 s average under reward v4), which the instantaneous state lacks.
        """
        parts = [full]
        if self.cfg["critic_scored_speed"]:
            parts.append((self.vx_avg[idx] / max(self.cfg["obs_cmd_scale"], 1e-3))[:, None])
        if self.cfg["critic_dr_params"]:
            rows = []
            for i in idx:
                p = self.dr_params(i)
                p["delay_ms"] /= 20.0                # roughly 0..1 like the others
                p["rail_height_mm"] /= 10.0
                rows.append(np.fromiter(p.values(), dtype=np.float64))
            parts.append(np.array(rows))
        return np.concatenate(parts, axis=1)

    def _obs_td(self) -> TensorDict:
        self.obs = np.nan_to_num(self.obs, nan=0.0, posinf=0.0, neginf=0.0)
        self.critic_obs = np.nan_to_num(self.critic_obs, nan=0.0, posinf=0.0, neginf=0.0)
        return TensorDict({"policy": torch.from_numpy(self.obs),
                           "critic": torch.from_numpy(self.critic_obs)},
                          batch_size=[self.num_envs])

    def _obs_full_many(self, idx: np.ndarray) -> np.ndarray:
        """Full 115-value simulator state for envs ``idx`` (critic, or actor with obs_mode='full')."""
        datas = [self.datas[i] for i in idx]
        qpos = np.array([d.qpos for d in datas])
        qvel = np.array([d.qvel for d in datas])
        quat = qpos[:, self.ball_qadr[:, None] + np.arange(4)]            # (k, 13, 4)
        rotvec = _quat_to_rotvec(quat).reshape(len(idx), -1)
        qvel_ball = qvel[:, self.ball_dadr[:, None] + np.arange(3)].reshape(len(idx), -1)
        ten = (self.l0 - np.array([d.ten_length for d in datas])) / self.stroke
        force = -np.array([d.actuator_force for d in datas]) / self.fmax
        ctrl = (self.l0 - np.array([d.ctrl for d in datas])) / self.stroke
        cp = self._cmd_phase(idx)
        return np.concatenate([
            (qpos[:, self.slide_qadr] / self.cfg["rail_half"])[:, None],
            (qvel[:, 0] / 0.20)[:, None],
            cp,
            rotvec,
            qvel_ball * 0.05,
            np.clip(ten, -0.5, 1.5),
            np.clip(force, 0.0, 1.5),
            np.clip(ctrl, 0.0, 1.0),
            self.prev_action[idx],
        ], axis=1)

    # ------------------------------------------------------------------ #
    def _draw(self, name: str, nominal: float = 1.0, size: int | None = None):
        """Sample ``dr_<name>`` (lo, hi), pulled toward nominal by dr_scale.

        An entry in ``dr_override`` fixes the value; with randomization off
        everything else stays nominal.
        """
        override = self.cfg["dr_override"]
        if name in override:
            value = float(override[name])
            return value if size is None else np.full(size, value)
        if not self.cfg["dr_enabled"]:
            return nominal if size is None else np.full(size, float(nominal))
        lo, hi = self.cfg["dr_" + name]
        u = self.rng.uniform(lo, hi, size)
        return nominal + self.cfg["dr_scale"] * (u - nominal)

    def _randomize(self, i: int) -> None:
        """Draw this episode's physics, servo, timing and sensor parameters."""
        self.battery[i] = self._draw("battery")
        self.sag[i] = self._draw("sag", 0.0)
        self.delay[i] = int(round(float(self._draw("delay", 0.0))))
        self.strength[i] = np.append(np.atleast_1d(self._draw("servo_strength", 1.0, self.n_servo)), 1.0)[self.servo_idx]
        c = self.cfg["dr_servo_calib"] * self.cfg["dr_scale"]
        if "servo_calib" in self.cfg["dr_override"]:
            self.calib[i] = float(self.cfg["dr_override"]["servo_calib"])
        elif self.cfg["dr_enabled"]:
            self.calib[i] = self.rng.uniform(-c, c, N_TENDON)
        else:
            self.calib[i] = 0.0
        if self.cfg["dr_enabled"]:
            s = self.cfg["dr_scale"]
            self.acc_bias[i] = self.rng.uniform(-1, 1, 3) * self.cfg["dr_acc_bias"] * s
            self.gyro_bias[i] = self.rng.uniform(-1, 1, 3) * self.cfg["dr_gyro_bias"] * s
            self.load_scale[i] = np.append(np.atleast_1d(self._draw("load_scale", 1.0, self.n_servo)), 1.0)[self.servo_idx]
        self.push_left[i] = 0
        self.volt[i] = self.battery[i]
        if not self.per_env_models:
            return

        mi, nom = self.models[i], self.nom
        for key, value in nom.items():                 # back to nominal first
            getattr(mi, key)[:] = value
        mi.geom_friction[[self.floor_geom, self.foot_geom], 0] = self._draw(
            "friction", float(nom["geom_friction"][self.foot_geom, 0]))
        mi.dof_frictionloss[self.slide_dof] *= self._draw("rail_friction")
        mi.dof_damping[self.slide_dof] *= self._draw("rail_damping")
        mi.body_pos[self.slider_body, 2] += self._draw("rail_height", 0.0)
        k = self._draw("carriage_mass")
        mi.body_mass[self.slider_body] *= k
        mi.body_inertia[self.slider_body] *= k
        k = self._draw("segment_mass", 1.0, len(self.leg_bodies))
        mi.body_mass[self.leg_bodies] *= k
        mi.body_inertia[self.leg_bodies] *= k[:, None]
        mi.jnt_stiffness[self.ball_jnt] *= self._draw("stiffness")
        mi.dof_damping[self.ball_dofs] *= self._draw("joint_damping")
        k = self._draw("servo_kp")
        mi.actuator_gainprm[:, 0] *= k
        mi.actuator_biasprm[:, 1] *= k
        mi.actuator_ctrlrange[:] = nom["actuator_ctrlrange"] + self.calib[i][:, None]
        mi.actuator_forcerange[:, 0] = nom["actuator_forcerange"][:, 0] * self.strength[i] * self.volt[i]

    def dr_params(self, i: int) -> dict:
        """This episode's randomized values (nominal = 1, or 0 for sag/delay)."""
        mi, nom = self.models[i], self.nom
        return {
            "friction": float(mi.geom_friction[self.foot_geom, 0]),
            "rail_friction": float(mi.dof_frictionloss[self.slide_dof] / nom["dof_frictionloss"][self.slide_dof]),
            "rail_damping": float(mi.dof_damping[self.slide_dof] / nom["dof_damping"][self.slide_dof]),
            "rail_height_mm": float(1000 * (mi.body_pos[self.slider_body, 2]
                                            - nom["body_pos"][self.slider_body, 2])),
            "carriage_mass": float(mi.body_mass[self.slider_body] / nom["body_mass"][self.slider_body]),
            "stiffness": float(mi.jnt_stiffness[self.ball_jnt[0]] / nom["jnt_stiffness"][self.ball_jnt[0]]),
            "joint_damping": float(mi.dof_damping[self.ball_dofs[0]] / nom["dof_damping"][self.ball_dofs[0]]),
            "servo_kp": float(mi.actuator_gainprm[0, 0] / nom["actuator_gainprm"][0, 0]),
            "servo_strength": float(self.strength[i].mean()),
            "battery": float(self.battery[i]),
            "sag": float(self.sag[i]),
            "delay_ms": float(self.delay[i] * self.m.opt.timestep * 1000),
            "volt": float(self.volt[i]),
        }

    def _reset_one(self, i: int, randomize: bool = True) -> None:
        self._randomize(i)
        mi = self.models[i]
        d = self.datas[i]
        d.xfrc_applied[:] = 0.0
        d.qpos[:] = self.qpos_straight
        d.qvel[:] = 0.0
        # Randomise which way the slight initial bend leans, so the policy
        # learns to push in both directions from the first update.
        bend = self.cfg["reset_bend"]
        if randomize and self.rng.random() < 0.5:
            bend = -bend
        half = 0.5 * bend
        # bend in the rail's plane (about the world y axis), whichever way the
        # leg is turned (leg_yaw_deg); the joint frames turn with the leg
        yaw = math.radians(self.cfg["leg_yaw_deg"])
        bend_q = np.array([math.cos(half), math.sin(half) * math.sin(yaw),
                           math.sin(half) * math.cos(yaw), 0.0])
        for adr in self.ball_qadr:
            d.qpos[adr:adr + 4] = bend_q

        noise = self.cfg["reset_noise"] if randomize else 0.0
        if noise > 0.0:
            for adr in self.ball_qadr:
                n = np.array([
                    1.0,
                    self.rng.normal(0.0, 0.003 * noise),
                    self.rng.normal(0.0, 0.003 * noise),
                    self.rng.normal(0.0, 0.003 * noise),
                ])
                n /= np.linalg.norm(n)
                w1, x1, y1, z1 = d.qpos[adr:adr + 4]
                w2, x2, y2, z2 = n
                q = np.array([
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ])
                d.qpos[adr:adr + 4] = q / np.linalg.norm(q)
        mujoco.mj_forward(mi, d)
        tip = d.site_xpos[self.tip_id]
        # put the foot near the rail centre; the hip moves the other way
        d.qpos[self.slide_qadr] = -tip[0] + self.rng.uniform(-0.03, 0.03) * noise
        d.qvel[0] = self.rng.uniform(-0.02, 0.02) * noise

        if self.cfg["fixed_pull_bias"]:
            pull = self.pull_bias_nominal
        else:  # legacy: pull that holds this episode's bent pose (sim-only)
            d.ctrl[:] = self.l0
            mujoco.mj_forward(mi, d)
            pull = np.clip((self.l0 - d.ten_length) / self.stroke, 0.0, 1.0)
        self.pull_bias[i] = pull
        hold = self.l0 - self.stroke * pull          # what the controller sends
        self.servo_cmd[i] = hold
        self.cmd_buf[i] = hold
        d.ctrl[:] = hold + self.calib[i]             # where the servo physically goes
        mujoco.mj_forward(mi, d)

        if self.cfg["reset_settle"] > 0:
            for _ in range(int(self.cfg["reset_settle"])):
                mujoco.mj_step(mi, d)
            d.ctrl[:] = hold + self.calib[i]

        self.t[i] = 0
        self.prev_action[i] = 0.0
        self.prev_action2[i] = 0.0
        self.prev_tension[i] = -d.actuator_force
        self.vx_avg[i] = float(d.qvel[0])
        self.vx_idle[i] = float(d.qvel[0])
        self.cmd_mag[i] = (self.rng.uniform(self.cfg["cmd_min"], self.cfg["cmd_max"])
                           if randomize else self.cfg["cmd_min"])
        self.cmd_sign0[i] = 1.0 if self.rng.random() < 0.5 else -1.0
        self.cmd_segment[i] = 0
        jitter = self.cfg["cmd_period_jitter"]
        self.cmd_period[i] = self.cfg["cmd_period"] * (1.0 + self.rng.uniform(-jitter, jitter))
        self.cmd[i] = 0.0  # the carriage starts at rest; a ramped command starts there too
        self._set_cmd(i)

    def _curriculum(self) -> float:
        """Cost multiplier k = k0^(kd^j), j = PPO iteration (Hwangbo et al. 2019).

        Starts at k0 and rises towards 1, so the policy first learns to move and
        then learns to move gently.  k0 = 0.05, kd = 0.997 gives k = 0.30 at
        iteration 300 and 0.86 at iteration 1000.  (The paper used k0 = 0.3;
        lower here because the rate/jerk costs also charge PPO's exploration
        noise, which made failing early attractive at the start.)
        """
        j = self.env_steps / max(self.cfg["curriculum_iter_steps"], 1)
        return float(self.cfg["curriculum_k0"] ** (self.cfg["curriculum_kd"] ** j))

    def _reward_one(self, i: int, action: np.ndarray, action_raw: np.ndarray):
        """Reward, failure flag and log values for env i (terms: rewards.py)."""
        d = self.datas[i]
        vx = float(d.qvel[0])
        x = float(d.qpos[self.slide_qadr])
        cmd = float(self.cmd[i])
        tip = d.site_xpos[self.tip_id]
        foot_z = float(tip[2])
        foot_y = float(tip[1])
        quat = np.stack([d.qpos[a:a + 4] for a in self.ball_qadr])
        bend = _bend_angle(quat)

        # The speed the velocity terms score; the critic sees it too (_obs_critic).
        tau = self.cfg["vel_avg_tau"]
        if tau > 0.0:
            self.vx_avg[i] += min(1.0, self.dt / tau) * (vx - self.vx_avg[i])
        else:
            self.vx_avg[i] = vx
        v_track = float(self.vx_avg[i])
        tau_idle = self.cfg["idle_tau"]
        self.vx_idle[i] += (min(1.0, self.dt / tau_idle) if tau_idle > 0.0 else 1.0) * (vx - self.vx_idle[i])
        slip = float(self.slip_sum[i] / self.slip_n[i]) if self.slip_n[i] > 0 else 0.0

        state = rewards.StepState(
            t=int(self.t[i]), cmd=cmd, vx=vx, v_track=v_track, v_idle=float(self.vx_idle[i]), x=x,
            foot_z=foot_z, foot_y=foot_y, bend=bend,
            pull=(self.l0 - d.ctrl) / self.stroke,
            force=-d.actuator_force / self.fmax,
            tension=-d.actuator_force, prev_tension=self.prev_tension[i],
            peak_tension=self.peak_tension[i], fmax=self.fmax,
            action=action, action_raw=action_raw,
            prev_action=self.prev_action[i], prev_action2=self.prev_action2[i],
            foot_force_peak=float(self.peak_foot_force[i]), slip=slip,
        )
        reward, parts, clipped = rewards.total(state, self.cfg, self._k)
        rail_half = self.cfg["rail_half"]
        failed = (
            foot_z > self.cfg["foot_lift_limit"]
            or bend.max() > self.cfg["bend_limit"]
            or abs(x) > rail_half * 0.98
            or not np.isfinite(d.qpos).all()
        )
        if failed:
            reward -= self.cfg["failure_penalty"]
        info = {
            "vx": vx, "v_track": v_track, "x": x, "cmd": cmd, "err": abs(v_track - cmd),
            "foot_z": foot_z, "foot_y": foot_y, "bend": float(bend.max()),
            **{f"r_{name}": value for name, value in parts.items()},
            "tension_peak": float(self.peak_tension[i].max()),
            "foot_force_peak": float(self.peak_foot_force[i]), "slip": slip,
            "clipped": float(clipped),
        }
        return float(reward), bool(failed), info

    # ------------------------------------------------------------------ #
    def _foot_contact(self, d, f6: np.ndarray, vel: np.ndarray) -> tuple[float, float]:
        """Total floor normal force and the fastest horizontal contact-point speed."""
        if d.ncon == 0:
            return 0.0, 0.0
        m = self.m
        fn = 0.0
        slip = 0.0
        geoms = d.contact.geom
        for c in range(d.ncon):
            g1, g2 = geoms[c]
            if g1 != self.floor_id and g2 != self.floor_id:
                continue
            mujoco.mj_contactForce(m, d, c, f6)
            fn += f6[0]
            g = g2 if g1 == self.floor_id else g1
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, vel, 0)
            # velocity of the contact point = v_geom + w x (p_contact - p_geom)
            rx, ry, rz = d.contact.pos[c] - d.geom_xpos[g]
            wx, wy, wz = vel[0], vel[1], vel[2]
            vx = vel[3] + wy * rz - wz * ry
            vy = vel[4] + wz * rx - wx * rz
            slip = max(slip, math.hypot(vx, vy))
        return fn, slip

    def _foot_sensor(self, d) -> tuple[float, float]:
        """Fast version of _foot_contact from the foot sensors (same quantities).

        The foot is a sphere, so the contact point slides at
        v_centre + w x (0, 0, -R) = (vx - R wy, vy + R wx).
        """
        s = d.sensordata
        r = self.foot_radius
        fn = float(s[self.foot_force_adr])
        vx, vy = s[self.foot_linvel_adr], s[self.foot_linvel_adr + 1]
        wx, wy = s[self.foot_angvel_adr], s[self.foot_angvel_adr + 1]
        return fn, math.hypot(vx - r * wy, vy + r * wx)

    def _step_chunk(self, idxs: np.ndarray) -> None:
        """One physics substep for some envs.  Only C code runs here, so the
        threads really run in parallel (mj_step releases Python's lock)."""
        models, datas = self.models, self.datas
        for i in idxs:
            mujoco.mj_step(models[i], datas[i])

    def _physics(self, cmd_buf: np.ndarray, push: np.ndarray) -> None:
        """n_substeps of physics for every env: servo model, then mj_step.

        The servo model (voltage sag, control delay, top speed) runs once per
        substep for all envs as array maths; only mj_step runs on the threads.
        (Running the per-env Python on the threads made them queue on Python's
        lock: 3 threads were no faster than 1.)
        """
        n, datas, models = self.num_envs, self.datas, self.models
        speed = self.cfg["servo_speed"]
        step_dl = speed * self.m.opt.timestep if speed > 0 else np.inf
        fmax_nom = self.nom["actuator_forcerange"][:, 0]          # negative (pull)
        track_foot = self.cfg["impact_weight"] != 0.0 or self.cfg["slip_weight"] != 0.0
        # physical setpoints: what was commanded, shifted by calibration error
        sent = cmd_buf + self.calib[:, None, :]                   # (n, [now, 1 ago, 2 ago], tendon)
        delay = self.delay.astype(np.int64)
        rows = np.arange(n)
        for i in range(n):
            datas[i].xfrc_applied[self.slider_body, 0] = push[i]
        # per-substep peaks for the reward (tension shock/limit, foot impact, slip)
        peak = np.full((n, N_TENDON), -np.inf)
        f_peak = np.zeros(n)
        slip_sum = np.zeros(n)
        slip_n = np.zeros(n, dtype=np.int32)
        force = np.array([d.actuator_force for d in datas])
        ctrl = np.array([d.ctrl for d in datas])
        volt = np.ones(n)
        r = self.foot_radius
        fa, la, wa = self.foot_force_adr, self.foot_linvel_adr, self.foot_angvel_adr
        for k in range(self.n_substeps):
            if self.per_env_models:
                # voltage sag: lower battery and heavy total load make every
                # servo weaker and slower
                load = np.mean(np.clip(-force / self.fmax, 0.0, 1.0), axis=1)
                volt = self.battery * (1.0 - self.sag * load)
                limit = fmax_nom[None, :] * self.strength * volt[:, None]
                for i in range(n):
                    models[i].actuator_forcerange[:, 0] = limit[i]
            # control delay: a command sent at step t reaches the servo `delay`
            # substeps later, so early substeps still chase older commands
            lag = np.where(k >= delay, 0, (delay - k + self.n_substeps - 1) // self.n_substeps)
            target = sent[rows, np.minimum(lag, CMD_BUF - 1)]
            max_dl = (step_dl * volt)[:, None]
            ctrl += np.clip(target - ctrl, -max_dl, max_dl)       # top speed
            for i in range(n):
                datas[i].ctrl[:] = ctrl[i]
            list(self._pool.map(self._step_chunk, self._chunks))
            force = np.array([d.actuator_force for d in datas])
            np.maximum(peak, -force, out=peak)
            if track_foot:
                s = np.array([d.sensordata for d in datas])
                fn = s[:, fa]
                slip = np.hypot(s[:, la] - r * s[:, wa + 1], s[:, la + 1] + r * s[:, wa])
                np.maximum(f_peak, fn, out=f_peak)
                touch = fn > 1e-6
                slip_sum += np.where(touch, slip, 0.0)
                slip_n += touch
        self.volt[:] = volt
        self.peak_tension[:] = peak
        self.peak_foot_force[:] = f_peak
        self.slip_sum[:] = slip_sum
        self.slip_n[:] = slip_n

    def step(self, actions, auto_reset: bool = True):
        action_raw = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_raw = action_raw.astype(np.float64)
        action = np.clip(action_raw, -1.0, 1.0)
        if action.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"actions shape {action.shape}, expected {(self.num_envs, self.num_actions)}")

        # each servo's action drives every cable on its spool
        target_pull = np.clip(
            self.pull_bias + self.cfg["residual_scale"]
            * np.concatenate([action, np.zeros((len(action), 1))], axis=1)[:, self.servo_idx], 0.0, 1.0)
        target_ctrl = self.l0[None, :] - self.stroke[None, :] * target_pull
        # controller-side smoothing of the sent target, then the servos chase it
        self.servo_cmd = np.clip(0.5 * self.servo_cmd + 0.5 * target_ctrl,
                                 self.ctrl_lo, self.l0)
        self.cmd_buf = np.roll(self.cmd_buf, 1, axis=1)
        self.cmd_buf[:, 0] = self.servo_cmd
        cmd_buf = self.cmd_buf.copy()

        # random carriage pushes (decided here so the RNG stays on this thread)
        if self.cfg["dr_enabled"]:
            p = self.cfg["dr_push_prob"] * self.cfg["dr_scale"]
            start = (self.push_left == 0) & (self.rng.random(self.num_envs) < p)
            if start.any():
                f = self.cfg["dr_push_force"]
                self.push_force[start] = self.rng.uniform(-f, f, int(start.sum()))
                self.push_left[start] = self.cfg["dr_push_steps"]
        push = np.where(self.push_left > 0, self.push_force, 0.0)
        self.push_left = np.maximum(self.push_left - 1, 0)

        self._physics(cmd_buf, push)

        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        failed = np.zeros(self.num_envs, dtype=bool)
        final_obs = np.zeros_like(self.obs)
        final_critic = np.zeros_like(self.critic_obs)
        log: dict[str, np.ndarray] = {}

        self._k = self._curriculum()
        noise = []
        for i in range(self.num_envs):
            self.t[i] += 1
            self._set_cmd(i)
            r, fail, info = self._reward_one(i, action[i], action_raw[i])
            reward[i] = r
            failed[i] = fail
            done[i] = fail or (self.t[i] >= self.max_episode_length)
            for k, v in info.items():
                if k not in log:
                    log[k] = np.zeros(self.num_envs)
                log[k][i] = v
            self.prev_action2[i] = self.prev_action[i]
            self.prev_action[i] = action[i]
            self.prev_tension[i] = -self.datas[i].actuator_force
            noise.append(self._obs_noise(i))        # random draws stay in env order
        self._push_obs_many(np.arange(self.num_envs), noise)
        final_obs[:] = self.obs
        final_critic[:] = self.critic_obs
        self.env_steps += 1

        time_out = done & ~failed
        if auto_reset and done.any():
            for i in np.flatnonzero(done):
                self._reset_one(i, randomize=True)
                self._push_obs(i, fresh=True)

        obs = self._obs_td()
        extras = {
            "time_outs": torch.from_numpy(time_out.astype(np.float32)),
            # both groups: LegPPO bootstraps timeouts with the critic's view
            "final_observation": TensorDict({
                "policy": torch.from_numpy(
                    np.nan_to_num(final_obs, nan=0.0, posinf=0.0, neginf=0.0)),
                "critic": torch.from_numpy(
                    np.nan_to_num(final_critic, nan=0.0, posinf=0.0, neginf=0.0)),
            }, batch_size=[self.num_envs]),
            "log": {
                "/reward_mean": float(reward.mean()),
                "/vx_mean": float(log["vx"].mean()),
                "/cmd_mean": float(log["cmd"].mean()),
                "/track_err": float(log["err"].mean()),
                "/x_mean": float(log["x"].mean()),
                "/x_abs_max": float(np.abs(log["x"]).max()),
                "/foot_z_mean": float(log["foot_z"].mean()),
                "/foot_y_mean": float(np.abs(log["foot_y"]).mean()),
                "/bend_max": float(log["bend"].max()),
                "/failed": float(failed.mean()),
                # every reward term, weighted (they sum to the reward before the
                # bounded-cost cap and the failure penalty)
                **{f"/{k}": float(v.mean()) for k, v in log.items() if k.startswith("r_")},
                "/v_track_mean": float(log["v_track"].mean()),
                "/slip": float(log["slip"].mean()),
                "/clipped": float(log["clipped"].mean()),
                "/curriculum_k": self._k,
                "/tension_peak_max": float(log["tension_peak"].max()),
                "/foot_force_peak_max": float(log["foot_force_peak"].max()),
            },
        }
        return (
            obs,
            torch.from_numpy(reward),
            torch.from_numpy(done.astype(np.float32)),
            extras,
        )

    # ------------------------------------------------------------------ #
    def reset(self):
        for i in range(self.num_envs):
            self._reset_one(i, randomize=True)
            self._push_obs(i, fresh=True)
        return self._obs_td()

    def get_observations(self):
        return self._obs_td()

    @property
    def episode_length_buf(self):
        return torch.from_numpy(self.t.copy())

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.t[:] = value.detach().cpu().numpy().astype(np.int32)

    def close(self):
        self._pool.shutdown(wait=True)
