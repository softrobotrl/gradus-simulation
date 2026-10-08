"""Double pendulum on a cart, Genesis + rsl-rl (>= 5.x).

DOF order: [cart (prismatic, actuated), shoulder (revolute, passive), elbow (revolute, passive)]
Angle convention: elbow = 0 means link 2 is collinear with link 1; "upright" is the
configuration where both links point straight up.
"""
import math
import xml.etree.ElementTree as ET

import torch
import genesis as gs
from tensordict import TensorDict

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_CFG = dict(
    urdf="robot.urdf",
    cart_joint="cart",      # joint names as Genesis reports them
    joint1="shoulder",
    joint2="elbow",
    sim_dt=0.01,            # physics step, same as free_swing_test.py
    substeps=2,             # same as free_swing_test.py
    decimation=2,           # physics steps per policy step -> control at 50 Hz
    x_limit=None,           # None -> derived from the URDF slider limits
    x_margin=0.02,          # episode ends this far (m) before the physical stop
    max_force=None,         # None -> URDF effort limit of the cart joint
    episode_s=10.0,
    down_at_zero=True,      # True if shoulder = 0 means hanging straight down in your CAD
    start="upright",        # "upright" (balance) | "mixed" (curriculum) | "down" | "random"
    init_noise=0.05,
    joint_damping=None,     # e.g. 0.0 to override damping coming from the URDF
    link_lengths=(1.0, 1.0),  # only the ratio matters (used for the tip-height term)
    reward_scales=None,     # partial overrides of DEFAULT_REWARD_SCALES
)

DEFAULT_REWARD_SCALES = dict(
    height=2.0,        # Increased tip height signal
    upright=5.0,       # Stronger peak when balanced
    cart_pos=0.5,      # Stronger centering force
    cart_edge=2.0,     # Penalize getting close to limits
    cart_vel=0.05,     
    ang_vel=0.005,     # Small un-gated damping penalty to discourage endless spinning
    action_rate=0.01,  
    action=0.005,      
    termination=5.0,  
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def wrap_angle(x):
    """Wrap to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def compute_reward(q, qd, actions, last_actions, fail, x_limit, up, scales, link_lengths):
    """Reward for balancing / swinging up a double pendulum on a cart.

    q, qd:        (N, 3) [cart, shoulder, elbow] positions / velocities
    actions:      (N, 1) current action in [-1, 1];  last_actions: previous action
    fail:         (N,) bool, True if the cart hit the rail end this step
    up:           shoulder angle at which the pendulum points straight up
    Returns (reward (N,), dict of weighted per-term tensors for logging).

    Design:
      * height     dense shaped term, tells the policy "higher tip is better" from any state.
      * upright    exp(-err^2) bonus, gives a sharp peak at the balanced pose so the policy
                   settles at the top instead of hovering near it.
      * ang_vel    penalized only near upright (gate), so fast swings are free during
                   swing-up but the policy must calm down once it arrives.
      * cart terms keep the cart centered and away from the rail ends.
      * action terms keep the control smooth.
    """
    l1, l2 = link_lengths
    # deviations from upright for the absolute angle of each link (wrapped)
    e1 = wrap_angle(q[:, 1] - up)
    e2 = wrap_angle(q[:, 1] + q[:, 2] - up)

    # normalized tip height in [-1, 1]; +1 = both links straight up
    h = (l1 * torch.cos(e1) + l2 * torch.cos(e2)) / (l1 + l2)

    height = h  

    # Continuous exponential bonus for both angles being upright
    upright = torch.exp(- (e1.pow(2) + e2.pow(2)) / 0.1)
    gate = upright                                            # "close to upright" weight

    xn = q[:, 0] / x_limit
    edge = torch.clamp((xn.abs() - 0.7) / 0.3, 0.0, 1.0)

    terms = dict(
        height=scales["height"] * height,
        upright=scales["upright"] * upright,
        cart_pos=-scales["cart_pos"] * xn ** 2,
        cart_edge=-scales["cart_edge"] * edge ** 2,
        cart_vel=-scales["cart_vel"] * qd[:, 0] ** 2,
        ang_vel=-scales["ang_vel"] * gate * (qd[:, 1] ** 2 + qd[:, 2] ** 2),
        action_rate=-scales["action_rate"] * (actions[:, 0] - last_actions[:, 0]) ** 2,
        action=-scales["action"] * actions[:, 0] ** 2,
        termination=-scales["termination"] * fail.float(),
    )
    rew = sum(terms.values())
    return rew, terms


def _urdf_joint_info(path, joint_name):
    """Return (lower, upper, effort) for a joint from the URDF, or None."""
    root = ET.parse(path).getroot()
    for j in root.iter("joint"):
        if j.get("name") in (joint_name, "dof_" + joint_name):
            lim = j.find("limit")
            if lim is None:
                return None
            lo, hi, eff = lim.get("lower"), lim.get("upper"), lim.get("effort")
            return (
                float(lo) if lo is not None else None,
                float(hi) if hi is not None else None,
                float(eff) if eff is not None else None,
            )
    return None


def _dof(robot, name):
    j = robot.get_joint(name)
    idx = getattr(j, "dofs_idx_local", None)
    return idx[0] if idx is not None else j.dof_idx_local  # API differs across versions


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class DoublePendulumCartEnv:
    def __init__(self, num_envs, cfg=None, show_viewer=False):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        c = self.cfg
        self.reward_scales = {**DEFAULT_REWARD_SCALES, **(c["reward_scales"] or {})}

        # rail limit and force limit straight from the URDF unless overridden
        info = _urdf_joint_info(c["urdf"], c["cart_joint"])
        if c["x_limit"] is None:
            if info is None or info[0] is None or info[1] is None:
                raise ValueError(
                    f"Could not read slider limits for '{c['cart_joint']}' from {c['urdf']}; "
                    "set x_limit in the config."
                )
            self.x_limit = min(-info[0], info[1]) - c["x_margin"]
        else:
            self.x_limit = c["x_limit"]
        if c["max_force"] is None:
            self.max_force = info[2] if (info and info[2] is not None) else 10.0
        else:
            self.max_force = c["max_force"]
        print(f"[env] x_limit={self.x_limit:.4f} m, max_force={self.max_force} N")

        self.num_envs = num_envs
        self.num_obs = 8
        self.num_actions = 1
        self.device = gs.device
        self.dt = c["sim_dt"] * c["decimation"]  # control period
        self.max_episode_length = int(c["episode_s"] / self.dt)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=c["sim_dt"], substeps=c["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -2.0, 1.5), camera_lookat=(0.0, 0.0, 0.8), refresh_rate=60
            ),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
            show_FPS=False,
        )
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=c["urdf"], fixed=True))
        self.scene.build(n_envs=num_envs)

        self.dofs = [_dof(self.robot, c[k]) for k in ("cart_joint", "joint1", "joint2")]
        self.cart_dof = self.dofs[:1]

        if c["joint_damping"] is not None:
            d = c["joint_damping"]
            d = list(d) if isinstance(d, (list, tuple)) else [d] * 3  # [cart, shoulder, elbow]
            self.robot.set_dofs_damping(
                torch.tensor(d, dtype=torch.float, device=self.device), self.dofs
            )

        # shoulder angle at which the pendulum is upright
        self.up = math.pi if c["down_at_zero"] else 0.0

        z = lambda *s: torch.zeros(*s, device=self.device)
        self.obs_buf = z(num_envs, self.num_obs)
        self.rew_buf = z(num_envs)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.actions = z(num_envs, 1)
        self.last_actions = z(num_envs, 1)
        self.extras = {}
        self.reset()  # valid observations before the runner first reads them

    # ---- helpers -------------------------------------------------------
    def _state(self):
        return self.robot.get_dofs_position(self.dofs), self.robot.get_dofs_velocity(self.dofs)

    def _compute_obs(self, q, qd):
        self.obs_buf = torch.stack(
            [
                q[:, 0] / self.x_limit, qd[:, 0],
                torch.sin(q[:, 1]), torch.cos(q[:, 1]),
                torch.sin(q[:, 2]), torch.cos(q[:, 2]),
                qd[:, 1], qd[:, 2],
            ],
            dim=1,
        )

    # ---- rsl-rl interface ----------------------------------------------
    def reset_idx(self, idx):
        if len(idx) == 0:
            return
        c, n = self.cfg, len(idx)
        noise = lambda s: (torch.rand(n, device=self.device) * 2 - 1) * s
        q = torch.zeros(n, 3, device=self.device)
        q[:, 0] = noise(0.3 * self.x_limit)
        mode = c["start"]
        if mode == "upright":
            q[:, 1] = self.up + noise(c["init_noise"])
            q[:, 2] = noise(c["init_noise"])
        elif mode == "mixed":
            # curriculum: half the envs start near upright (wide noise), half start anywhere
            near = torch.rand(n, device=self.device) < 0.5
            q[:, 1] = torch.where(near, self.up + noise(0.3), noise(math.pi))
            q[:, 2] = torch.where(near, noise(0.3), noise(1.0))
        elif mode == "down":
            q[:, 1] = (self.up + math.pi) + noise(c["init_noise"])
            q[:, 2] = noise(c["init_noise"])
        else:  # random
            q[:, 1] = noise(math.pi)
            q[:, 2] = noise(1.0)
        self.robot.set_dofs_position(
            position=q, dofs_idx_local=self.dofs, zero_velocity=True, envs_idx=idx
        )
        self.episode_length_buf[idx] = 0
        self.actions[idx] = 0.0
        self.last_actions[idx] = 0.0

    def step(self, actions):
        self.last_actions = self.actions.clone()
        self.actions = torch.clip(actions, -1.0, 1.0)
        self.robot.control_dofs_force(self.actions * self.max_force, dofs_idx_local=self.cart_dof)
        for _ in range(self.cfg["decimation"]):
            self.scene.step()
        self.episode_length_buf += 1

        q, qd = self._state()
        fail = q[:, 0].abs() > self.x_limit

        self.rew_buf, terms = compute_reward(
            q, qd, self.actions, self.last_actions, fail,
            self.x_limit, self.up, self.reward_scales, self.cfg["link_lengths"],
        )

        timeout = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = fail | timeout
        self.extras = {
            "time_outs": timeout.float(),
            "log": {f"/rew_{k}": v for k, v in terms.items()},
        }

        self.reset_idx(torch.nonzero(self.reset_buf).flatten())
        q, qd = self._state()
        self._compute_obs(q, qd)
        return self.get_observations(), self.rew_buf, self.reset_buf.long(), self.extras

    def get_observations(self):
        # rsl-rl >= 5: observation groups in a TensorDict; "policy" is mapped in obs_groups
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs], device=self.device)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        q, qd = self._state()
        self._compute_obs(q, qd)
        return self.get_observations()