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
    max_force=1,         # None -> URDF effort limit of the cart joint
    episode_s=10.0,
    down_at_zero=True,      # True if shoulder = 0 means hanging straight down in your CAD
    start="upright",        # "upright" (balance) | "mixed" (curriculum) | "down" | "random"
    init_noise=0.02,
    joint_damping=None,     # e.g. 0.0 to override damping coming from the URDF
    link_lengths=(1.0, 1.0),  # only the ratio matters (used for the tip-height term)
    reward_scales=None,     # partial overrides of DEFAULT_REWARD_SCALES
)

DEFAULT_REWARD_SCALES = dict(
    energy=1.0,        # Dense Lyapunov-like signal for swing-up
    upright=4.0,       # Sharp bonus for precise balancing
    cart_pos=0.1,      # Gentle centering
    action_rate=0.01,  # Smooth control
    action=0.005,      # Small effort penalty
)
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def wrap_angle(x):
    """Wrap to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))

def compute_reward(q, qd, actions, last_actions, fail, x_limit, up, scales, link_lengths):
    """
    Dense, Lyapunov-inspired reward for rapid swing-up learning.

    Design:
      - Primary term: "energy" of the pendulum system, which is lowest at upright.
      - Guidance terms: penalize cart offset and control effort to encourage efficient, stable motions.
    """
    l1, l2 = link_lengths
    m1 = m2 = 1.0  # Assume unit masses for the energy proxy; only ratios matter.

    # Angles relative to upright
    e1 = wrap_angle(q[:, 1] - up)
    e2 = wrap_angle(q[:, 1] + q[:, 2] - up)

    # ---- 1. Lyapunov-like energy term ----
    # Potential energy (normalized): lowest (most negative) when both links point up.
    # Kinetic energy (normalized): lowest (zero) at rest.
    potential = -(l1 * torch.cos(e1) + l2 * torch.cos(e2)) / (l1 + l2)  # in [-1, 1]
    kinetic = 0.5 * (m1 * (l1 * qd[:, 1])**2 + m2 * (l2 * (qd[:, 1] + qd[:, 2]))**2) / (m1 * l1**2 + m2 * l2**2)

    # The energy term is low (reward is high) only when potential is low AND kinetic is low.
    energy_reward = 1.0 - 0.5 * (potential**2 + kinetic**2)  # in [0, 1]

    # ---- 2. Upright bonus (sharp, gated) ----
    # This gives a strong pull to settle *exactly* at the top.
    angle_error_sq = e1.pow(2) + e2.pow(2)
    upright_bonus = torch.exp(-angle_error_sq / 0.05)

    # ---- 3. Cart centering ----
    x_normalized = q[:, 0] / x_limit
    cart_center_penalty = -x_normalized.pow(2)

    # ---- 4. Control effort ----
    action_magnitude_penalty = -actions[:, 0].pow(2)
    action_rate_penalty = -(actions[:, 0] - last_actions[:, 0]).pow(2)

    # ---- 5. Termination ----
    termination_penalty = -fail.float() * 10.0

    # ---- Weighted sum ----
    terms = {
        "energy": scales["energy"] * energy_reward,
        "upright_bonus": scales["upright"] * upright_bonus,
        "cart_center": scales["cart_pos"] * cart_center_penalty,
        "action_mag": scales["action"] * action_magnitude_penalty,
        "action_rate": scales["action_rate"] * action_rate_penalty,
        "termination": termination_penalty,
    }
    reward = sum(terms.values())
    return reward, terms

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
                camera_pos=(0.85, -1.55, 0.7), camera_lookat=(0.0, 0.0, 0.3), refresh_rate=60
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

        down = self.up + math.pi              # shoulder angle when hanging straight down
        near_top_scale = 0.10                 # tight noise around upright (was 0.3)

        mode = c["start"]
        if mode == "upright":
            q[:, 1] = self.up + noise(c["init_noise"])
            q[:, 2] = noise(c["init_noise"])
        elif mode == "mixed":
            # 50/50 split: half near upright (small noise), half anywhere in a full swing
            # around the DOWN position. Explicitly anchoring the far branch to `down`
            # guarantees it can never land on the upright pose.
            near = torch.rand(n, device=self.device) < 0.5
            q[:, 1] = torch.where(
                near,
                self.up + noise(near_top_scale),
                down + noise(math.pi),
            )
            q[:, 2] = torch.where(
                near,
                noise(near_top_scale),
                noise(math.pi),
            )
        elif mode == "down":
            q[:, 1] = down + noise(c["init_noise"])
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