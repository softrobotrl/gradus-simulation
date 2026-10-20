"""Two-tendon (computed) actuation of the Gradus ball-socket leg in Genesis.

The leg is the 16-disk ball-socket chain from the Onshape document. Two fishing
lines run through the 4 mm cable holes: one on the +X (front) side and one on
the -X (back) side. Instead of simulating the lines as soft bodies, we compute
each line's routed length L(q) from the joint angles and apply the generalized
tendon force

    tau = -(dL/dq) * F        (F >= 0, tension only)

for both tendons. This is the usual sim-to-real tendon model and it is
differentiable/fast enough for RL.

Run:
    python genesis_samples/gradus_tendon.py                 # headless, prints tracking
    python genesis_samples/gradus_tendon.py --viewer --xray # live viewer, see-through links
    python genesis_samples/gradus_tendon.py --render --xray # save frames to logs/gradus_tendon
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import genesis as gs

HERE = Path(__file__).resolve().parent.parent
URDF = HERE / "urdf" / "gradus" / "gradus_leg.urdf"

# From urdf/gradus/onshape_params.json
N_S1 = 8
N = 16
PITCH = 0.03125
HOLE_R = [0.0375] * N_S1 + [0.029166666666666667] * (N - N_S1)

BASE_ANCHOR_Z = 0.05  # actuator/spool sits this far above the hip
TIP_EXTENSION = -0.8 * PITCH

# antagonistic controller gains / limits
F_CO = 55.0  # co-contraction [N]
F_MAX = 220.0  # per-line tension limit [N]
KP = 70.0  # N per rad of bend error
KD = 8.0  # N per (rad/s) of bend rate


class ComputedTendons:
    """Two routed lines, one at +X and one at -X of the leg centreline."""

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.r = torch.tensor(HOLE_R, device=device, dtype=dtype)
        self.pitch = PITCH
        self.base_z = BASE_ANCHOR_Z
        self.tip_ext = TIP_EXTENSION

    def _origins(self, q: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """Joint origins in the base frame, (B, N, 3)."""
        b, n = q.shape
        origin = torch.zeros(b, n, 3, device=self.device, dtype=self.dtype)
        origin[:, 0, 2] = -self.pitch
        for i in range(1, n):
            a = ang[:, i - 1]
            origin[:, i, 0] = origin[:, i - 1, 0] - self.pitch * torch.sin(a)
            origin[:, i, 2] = origin[:, i - 1, 2] - self.pitch * torch.cos(a)
        return origin

    def _routed_points(self, q: torch.Tensor, side: float) -> torch.Tensor:
        """World (base-frame) points [anchor, guide_0..guide_15, tip_anchor]."""
        b, n = q.shape
        ang = torch.cumsum(q, dim=1)

        origin = self._origins(q, ang)
        c, s = torch.cos(ang), torch.sin(ang)
        guide = torch.empty_like(origin)
        guide[:, :, 0] = origin[:, :, 0] + side * self.r[None, :] * c
        guide[:, :, 1] = origin[:, :, 1]
        guide[:, :, 2] = origin[:, :, 2] - side * self.r[None, :] * s

        base = torch.zeros(b, 1, 3, device=self.device, dtype=self.dtype)
        base[:, 0, 0] = side * self.r[0]
        base[:, 0, 2] = self.base_z

        tip = torch.empty(b, 1, 3, device=self.device, dtype=self.dtype)
        tip[:, 0, 0] = origin[:, -1, 0] + side * self.r[-1] * c[:, -1]
        tip[:, 0, 1] = origin[:, -1, 1]
        tip[:, 0, 2] = origin[:, -1, 2] - side * self.r[-1] * s[:, -1] + self.tip_ext

        return torch.cat([base, guide, tip], dim=1)

    def length(self, q: torch.Tensor, side: float) -> torch.Tensor:
        pts = self._routed_points(q, side)
        return (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(dim=1)

    def points(self, q: torch.Tensor, side: float) -> torch.Tensor:
        return self._routed_points(q, side)

    def torques(self, q: torch.Tensor, tensions: torch.Tensor, retain_graph: bool = False):
        """tensions: (B, 2) = [front +X, back -X] -> joint torques (B, N)."""
        qv = q.detach().clone().requires_grad_(True)
        l_front = self.length(qv, +1.0)
        l_back = self.length(qv, -1.0)
        g_front = torch.autograd.grad(l_front.sum(), qv, retain_graph=True)[0]
        g_back = torch.autograd.grad(l_back.sum(), qv, retain_graph=retain_graph)[0]
        tau = -(g_front * tensions[:, 0:1] + g_back * tensions[:, 1:2])
        lengths = torch.stack([l_front.detach(), l_back.detach()], dim=1)
        return tau, lengths


class GradusLegTendonEnv:
    def __init__(
        self,
        num_envs: int = 1,
        show_viewer: bool = False,
        render: bool = False,
        xray: bool = False,
        dt: float = 0.005,
        substeps: int = 4,
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.tendon = ComputedTendons(gs.device, gs.tc_float)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=(0.0, 0.0, -9.81)),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=10,
                tolerance=1e-5,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -1.0, 0.55),
                camera_lookat=(0.0, 0.0, 0.47),
                camera_fov=38,
            ),
            show_viewer=show_viewer,
        )

        # visual reference only; the leg does not touch it in this demo
        self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(URDF),
                pos=(0.0, 0.0, 0.72),
                fixed=True,
                merge_fixed_links=False,
            ),
            surface=gs.surfaces.Default(opacity=0.35) if xray else None,
            name="gradus_leg",
        )

        self.cam = None
        if render:
            self.cam = self.scene.add_camera(
                res=(720, 720), pos=(0.0, -1.0, 0.55), lookat=(0.0, 0.0, 0.47), fov=38, GUI=False
            )

        self.scene.build(n_envs=num_envs)
        self.dofs = torch.arange(N, dtype=gs.tc_int, device=gs.device)

        # pure force control: no PD on the tendon-driven joints
        self.robot.set_dofs_kp(torch.zeros(N, device=gs.device))
        self.robot.set_dofs_kv(torch.zeros(N, device=gs.device))

        self.robot.set_dofs_position(torch.zeros(num_envs, N, device=gs.device))
        self.robot.zero_all_dofs_velocity()

        # sanity check: our analytic joint origins must match Genesis link frames
        ang = torch.cumsum(torch.zeros(1, N, device=gs.device), dim=1)
        pred = self.tendon._origins(torch.zeros(1, N, device=gs.device), ang)[0]
        pred[:, 2] += 0.72
        err = max(
            float(
                torch.linalg.norm(
                    self.robot.get_link(f"disk_{i}").get_pos().reshape(-1, 3)[0] - pred[i]
                )
            )
            for i in range(N)
        )
        print(f"[check] analytic FK vs Genesis link origins: max error {err * 1e3:.3f} mm")

    # ------------------------------------------------------------------ control
    def step(self, bend_des: torch.Tensor):
        q = self.robot.get_dofs_position(self.dofs)
        qd = self.robot.get_dofs_velocity(self.dofs)

        bend = q.sum(dim=1)
        rate = qd.sum(dim=1)
        err = bend_des - bend

        diff = KP * err - KD * rate  # N of differential tension
        f_back = torch.clamp(F_CO + diff, 0.0, F_MAX)  # -X tendon pulls q > 0
        f_front = torch.clamp(F_CO - diff, 0.0, F_MAX)  # +X tendon pulls q < 0
        tensions = torch.stack([f_front, f_back], dim=1)

        tau, lengths = self.tendon.torques(q, tensions)
        self.robot.control_dofs_force(tau, self.dofs)
        self.scene.step()

        return {
            "q": q,
            "qd": qd,
            "bend": bend,
            "bend_des": bend_des,
            "tensions": tensions,
            "lengths": lengths,
            "tau": tau,
        }

    # ------------------------------------------------------------------ visuals
    def draw_tendons(self):
        q = self.robot.get_dofs_position(self.dofs)
        self.scene.clear_debug_objects()
        for side, color in ((+1.0, (0.95, 0.25, 0.15, 1.0)), (-1.0, (0.15, 0.55, 0.95, 1.0))):
            pts = self.tendon.points(q, side)[0].cpu().numpy()
            pts[:, 2] += 0.72  # robot mount height
            for a, b in zip(pts[:-1], pts[1:]):
                self.scene.draw_debug_line(a, b, radius=0.0025, color=color)

    def render(self, path: str):
        if self.cam is None:
            return
        rgb = self.cam.render(rgb=True)[0]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        from PIL import Image

        Image.fromarray(rgb).save(path)


def plot_results(history, tendon, q_pose, out_dir):
    """Save the tracking plot and a side-view schematic of the two routed lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    t = np.asarray(history["t"])
    bend = np.asarray(history["bend"])
    des = np.asarray(history["des"])

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax.plot(t, des, "--", color="0.35", lw=1.6, label="commanded Σq")
    ax.plot(t, bend, color="#e24a33", lw=2.0, label="actual Σq")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("total bend [rad]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("two-tendon antagonistic tracking")

    # ---- side view of the routed lines -------------------------------------
    q = q_pose[None] if q_pose is not None else torch.zeros(1, N, dtype=tendon.dtype)
    ang = torch.cumsum(q, dim=1)[0].cpu().numpy()
    origin = tendon._origins(q, torch.cumsum(q, dim=1))[0].cpu().numpy()
    radius = tendon.r.cpu().numpy()

    axr.add_patch(Rectangle((-0.06, 0.0), 0.12, 0.06, fc="0.2", ec="none", zorder=2))
    for i in range(N):
        p = np.array([np.cos(ang[i]), -np.sin(ang[i])])
        o = origin[i][[0, 2]]
        axr.plot(
            [o[0] - radius[i] * p[0], o[0] + radius[i] * p[0]],
            [o[1] - radius[i] * p[1], o[1] + radius[i] * p[1]],
            color="#8fb8d8" if i < N_S1 else "0.72",
            lw=7,
            alpha=0.55,
            solid_capstyle="round",
            zorder=3,
        )
        if i + 1 < N:
            o2 = origin[i + 1][[0, 2]]
            axr.plot([o[0], o2[0]], [o[1], o2[1]], color="0.82", lw=1.0, zorder=1)

    for side, color, label in (
        (+1.0, "#e24a33", "front line (+X)"),
        (-1.0, "#348abd", "back line (-X)"),
    ):
        pts = tendon.points(q, side)[0].cpu().numpy()
        axr.plot(pts[:, 0], pts[:, 2], color=color, lw=1.8, zorder=5, label=label)
        axr.scatter(pts[1:-1, 0], pts[1:-1, 2], s=9, color=color, zorder=6)
        axr.scatter(pts[0, 0], pts[0, 2], s=28, marker="s", color=color, zorder=7)

    axr.set_aspect("equal")
    axr.set_xlim(-0.18, 0.18)
    axr.set_ylim(-0.55, 0.10)
    axr.set_xlabel("x [m]")
    axr.set_ylabel("z [m]")
    axr.grid(alpha=0.2)
    axr.legend(loc="lower left", frameon=False)
    axr.set_title(f"routing at max bend ({q.sum().item():+.2f} rad)")

    fig.tight_layout()
    path = out_dir / "results.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--xray", action="store_true", help="semi-transparent links so the tendon routing is visible")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")
    env = GradusLegTendonEnv(args.num_envs, args.viewer, args.render, args.xray)

    out_dir = HERE / "logs" / "gradus_tendon"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(args.seconds / env.dt)
    peak = 1.1  # rad of total bend
    i = 0
    log = []
    history = {k: [] for k in ("t", "bend", "des", "F_front", "F_back")}
    max_bend = 0.0
    q_pose = None
    for step in range(n_steps):
        t = step * env.dt
        # smooth commanded curl: 0 -> +peak -> 0 -> -peak -> 0
        bend_des = torch.full((args.num_envs,), peak * math.sin(2 * math.pi * 0.16 * t), device=gs.device)
        info = env.step(bend_des)

        bend_now = info["bend"][0].item()
        history["t"].append(t)
        history["bend"].append(bend_now)
        history["des"].append(bend_des[0].item())
        history["F_front"].append(info["tensions"][0, 0].item())
        history["F_back"].append(info["tensions"][0, 1].item())
        if abs(bend_now) > max_bend:
            max_bend = abs(bend_now)
            q_pose = info["q"][0].detach().clone()

        if step % 100 == 0:
            log.append((t, bend_now, bend_des[0].item(),
                        info["tensions"][0, 0].item(), info["tensions"][0, 1].item()))
            if args.viewer:
                env.draw_tendons()

        if args.render and step % 40 == 0:
            env.draw_tendons()
            env.render(str(out_dir / f"frame_{i:04d}.png"))
            i += 1

    print(f"{'t':>6} {'bend':>8} {'des':>8} {'F_front':>8} {'F_back':>8}")
    for row in log:
        print(f"{row[0]:6.2f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.1f} {row[4]:8.1f}")

    q = env.robot.get_dofs_position(env.dofs)[0].cpu().numpy()
    tip = env.robot.get_link("foot").get_pos().cpu().numpy().reshape(-1, 3)[0]
    qv = torch.tensor(q, dtype=gs.tc_float, device=gs.device)[None]
    lf = env.tendon.length(qv, +1.0).item()
    lb = env.tendon.length(qv, -1.0).item()
    l0 = env.tendon.length(torch.zeros_like(qv), +1.0).item()
    print(f"final bend={q.sum():+.3f} rad, tip=({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m")
    print(f"tendon length front={lf * 1e3:.1f} mm (straight {l0 * 1e3:.1f} mm), back={lb * 1e3:.1f} mm")

    plot_path = plot_results(history, env.tendon, q_pose, out_dir)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
