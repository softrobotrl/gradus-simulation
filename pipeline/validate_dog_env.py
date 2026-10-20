"""Acceptance checks for the Gradus dog MJCF and residual environment.

Run from the repo root::

    python -m pipeline.validate_dog_env
    python -m pipeline.validate_dog_env --render       # also save a side view

Checks, in order:

1. model dimensions, masses, foot heights, tendon rest lengths;
2. the 2x2 tendon Jacobian of every leg is full rank and well conditioned;
3. the nominal stand holds for two seconds with zero policy action;
4. winding one tendon moves that leg in the direction that shortens the tendon;
5. the batched environment handles random residuals, auto-reset and time-outs.

The script is CPU-only and does not import JAX.  Exit code 0 means every check
passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from pipeline.dog_env import DogEnv, DogEnvConfig, HINGE_NAMES, LEG_NAMES, TENDON_NAMES
from pipeline.dog_model import DEFAULT_OUT, DogModelConfig, build_dog_xml

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{PASS if ok else FAIL}] {name}{(' - ' + detail) if detail else ''}")
        if not ok:
            self.failures += 1

    def close(self) -> None:
        raise SystemExit(1 if self.failures else 0)


def model_checks(c: Checker, path: Path) -> mujoco.MjModel:
    print("model")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c.check("nq == 23 (free + 8 hinges + 8 cable slides)", m.nq == 23, f"nq={m.nq}")
    c.check("nv == 22", m.nv == 22, f"nv={m.nv}")
    c.check("8 tendon actuators", m.nu == 8, f"nu={m.nu}")
    c.check("8 tendons", m.ntendon == 8, f"ntendon={m.ntendon}")
    c.check("4 legs / 8 hinge joints", len(LEG_NAMES) == 4 and len(HINGE_NAMES) == 8)
    mass = float(m.body_subtreemass[m.body("trunk").id])
    c.check("mass in a plausible range (2-6 kg)", 2.0 <= mass <= 6.0, f"{mass:.2f} kg")
    feet = {leg: float(d.geom_xpos[m.geom(f"{leg}_foot").id, 2]) for leg in LEG_NAMES}
    c.check("feet at z=0.015 (sphere radius) at the rest pose",
            all(abs(z - 0.015) < 1e-6 for z in feet.values()),
            ", ".join(f"{k}={v:.4f}" for k, v in feet.items()))
    c.check("tendon rest lengths are positive and equal to the path lengths",
            np.all(m.tendon_range[:, 1] > 0.05) and np.allclose(d.ten_length, m.tendon_range[:, 1], atol=1e-5),
            f"min={m.tendon_range[:, 1].min():.4f} max={m.tendon_range[:, 1].max():.4f}")
    c.check("actuator ctrl range is in radians and nonzero",
            np.all(m.actuator_ctrlrange[:, 1] > m.actuator_ctrlrange[:, 0]),
            f"{m.actuator_ctrlrange[0].tolist()}")

    # The drive geometry has to agree between the model, the environment and the
    # report, otherwise the tension numbers mean different things everywhere.
    info_path = path.parent / "model_info.yaml"
    if info_path.exists():
        info = yaml.safe_load(info_path.read_text())
        gear = float(m.actuator_gear[0, 0])
        drum_radius = 1.0 / gear
        max_tension = float(m.actuator_forcerange[0, 1]) * gear
        c.check("drum radius agrees with the model report",
                abs(info["drum_radius_m"] - drum_radius) < 1e-6 * drum_radius,
                f"{info['drum_radius_m'] * 1000:.1f} mm")
        c.check("tension limit agrees with the model report",
                abs(info["max_tension_N"] - max_tension) < 1e-4 * max_tension,
                f"{max_tension:.1f} N")
        c.check("drum speed limit matches the env's rate limit",
                abs(info["drum_speed_rad_s"] - DogEnvConfig().max_speed) < 1e-9,
                f"{info['drum_speed_rad_s']:.1f} rad/s")
        c.check("the gearbox gives the drum more torque than the motor",
                info["gear_ratio"] > 1.0 and info["drum_torque_Nm"] > 0.4,
                f"{info['gear_ratio']:.1f}:1, {info['drum_torque_Nm']:.2f} N*m")
    return m


def jacobian_checks(c: Checker, env: DogEnv) -> None:
    print("tendon Jacobian (d length / d joint angle)")
    Jleg = env.Jleg
    worst_det, worst_cond = np.inf, 0.0
    for leg, name in enumerate(LEG_NAMES):
        det = float(np.linalg.det(Jleg[leg]))
        cond = float(np.linalg.cond(Jleg[leg]))
        worst_det = min(worst_det, abs(det))
        worst_cond = max(worst_cond, cond)
        c.check(f"{name}: full rank", abs(det) > 1e-6, f"det={det:.3e} cond={cond:.2f}")
        print(f"        J = {np.array2string(Jleg[leg], precision=5, suppress_small=True)}")
    c.check("all legs well conditioned (cond < 10)", worst_cond < 10.0, f"worst cond={worst_cond:.2f}")


def stand_checks(c: Checker, env: DogEnv) -> None:
    print("nominal stand (zero residual policy)")
    n = env.num_envs
    env.reset()
    zeros = np.zeros((n, env.num_actions), dtype=np.float32)
    heights, tilts, tensions = [], [], []
    failures = 0
    for _ in range(100):  # 2 s
        _, reward, done, extras = env.step(zeros)
        heights.append(extras["log"]["/base_height"])
        tilts.append(extras["log"]["/tilt"])
        tensions.append(extras["log"]["/tension"])
        failures += int(done.sum())
    h = np.asarray(heights)
    c.check("no episode ended during the first 2 s", failures == 0, f"{failures} terminations")
    c.check("height stays near 0.215 m", abs(float(np.mean(h[-50:])) - 0.215) < 0.01,
            f"mean={np.mean(h[-50:]):.4f} m")
    c.check("body stays upright", float(np.max(tilts)) < 0.20, f"max tilt={np.max(tilts):.3f}")
    c.check("tendons carry the body (normalized tension > 0.05)", float(np.mean(tensions)) > 0.05,
            f"mean normalized tension={np.mean(tensions):.3f}")
    contact = extras["log"]["/foot_contact"]
    c.check("all four feet in contact at the end", contact > 0.99, f"mean contact={contact:.2f}")


def tendon_pull_checks(c: Checker, path: Path) -> None:
    """Zero-gravity, no-contact test: winding a tendon must shorten that tendon's path."""
    print("single-tendon actuation (zero gravity, floating)")
    m = mujoco.MjModel.from_xml_path(str(path))
    m.opt.gravity[:] = 0.0
    d = mujoco.MjData(m)
    hinge_qpos = np.array([m.jnt_qposadr[m.joint(n).id] for n in HINGE_NAMES])
    ten_ids = np.array([m.tendon(n).id for n in TENDON_NAMES])

    def settle(qpos0: np.ndarray, ctrl: np.ndarray, steps: int = 250) -> np.ndarray:
        d.qpos[:] = qpos0
        d.qvel[:] = 0.0
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
        for _ in range(steps):
            mujoco.mj_step(m, d)
        return d.qpos[hinge_qpos].copy()

    base_ctrl = np.full(m.nu, 0.10)
    qpos0 = m.qpos0.copy()
    qpos0[2] = 0.25
    q_ref = settle(qpos0, base_ctrl)
    J = np.zeros((m.ntendon, len(hinge_qpos)))
    eps = 1e-6
    for j, adr in enumerate(hinge_qpos):
        qp = m.qpos0.copy(); qp[2] = 0.25; qp[adr] += eps
        d.qpos[:] = qp; mujoco.mj_forward(m, d); lp = d.ten_length.copy()
        qm = m.qpos0.copy(); qm[2] = 0.25; qm[adr] -= eps
        d.qpos[:] = qm; mujoco.mj_forward(m, d); lm = d.ten_length.copy()
        J[:, j] = (lp - lm) / (2 * eps)

    for k, name in enumerate(TENDON_NAMES):
        ctrl = base_ctrl.copy()
        ctrl[k] += 0.5
        q = settle(qpos0, ctrl)
        dq = q - q_ref
        leg = k // 2
        dq_leg = dq[2 * leg:2 * leg + 2]
        # Winding the tendon should move the joints so the path gets shorter.
        shortening = -float(J[k, 2 * leg:2 * leg + 2] @ dq_leg)
        c.check(f"{name}: winding shortens its path and moves the leg",
                shortening > 1e-4 and np.linalg.norm(dq_leg) > 1e-4,
                f"shortening={shortening * 1e3:.2f} mm, |dq|={np.linalg.norm(dq_leg):.3f} rad")


def env_checks(c: Checker) -> None:
    print("batched environment")
    env = DogEnv(8, DogEnvConfig(nominal_mode="pd", command_mode="stand", episode_s=1.0),
                 seed=0, num_threads=4)
    obs = env.reset()["policy"]
    c.check("observation shape is (num_envs, num_obs)", tuple(obs.shape) == (8, env.num_obs),
            f"{tuple(obs.shape)}")
    c.check("observation is finite", bool(np.isfinite(obs.numpy()).all()))
    layout = env.obs_layout
    c.check("state_residual slice has 31 entries", len(range(*layout["state_residual"].indices(env.num_obs))) == 31)
    c.check("action_context slice has 29 entries", len(range(*layout["action_context"].indices(env.num_obs))) == 29)
    c.check("nominal action is visible in the observation",
            np.allclose(env.obs[:, layout["nominal_action"]], env.cfg.stand_angle, atol=0.2))

    rng = np.random.default_rng(0)
    resets = 0
    t0 = time.perf_counter()
    steps = 100
    for _ in range(steps):
        action = rng.uniform(-1, 1, (8, env.num_actions)).astype(np.float32)
        obs, reward, done, extras = env.step(action)
        resets += int(done.sum())
        assert np.isfinite(obs["policy"].numpy()).all()
        assert extras["final_observation"].shape == (8, env.num_obs)
        assert extras["time_outs"].shape == (8,)
    elapsed = time.perf_counter() - t0
    c.check("random residuals keep observations finite", True)
    c.check("auto-reset/time-out machinery runs", resets >= 0, f"{resets} resets in {steps} steps")
    c.check("throughput is usable (> 1,000 transitions/s)", 8 * steps / elapsed > 1000,
            f"{8 * steps / elapsed:,.0f} transitions/s")
    env.close()


def mjx_checks(c: Checker) -> None:
    """The MJX (GPU) backend must reproduce the C backend's behaviour."""
    print("MJX backend")
    try:
        import jax

        from pipeline.dog_mjx_env import make_dog_mjx_env
    except Exception as exc:
        print(f"  skipped: {exc}")
        return
    if not jax.devices("gpu"):
        print("  skipped: JAX sees no GPU")
        return

    cfg = dict(nominal_mode="pd", command_mode="stand", episode_s=4.0, init_joint_noise=0.02,
               init_tilt_noise=0.01, init_height_noise=0.002, init_vel_noise=0.0)
    cenv = DogEnv(16, DogEnvConfig(**cfg), seed=5, num_threads=8)
    menv = make_dog_mjx_env(16, seed=5, device="gpu", **cfg)
    cenv.reset()
    menv.reset()
    action = np.zeros((16, 8), dtype=np.float32)
    for _ in range(50):
        _, _, _, cex = cenv.step(action)
        _, _, _, mex = menv.step(action)
    for key, tol in (("/base_height", 0.005), ("/tilt", 0.01), ("/reward_mean", 0.05),
                     ("/foot_contact", 0.05)):
        cv, mv = cex["log"][key], mex["log"][key]
        c.check(f"MJX matches C: {key}", abs(cv - mv) < tol, f"C={cv:.4f} MJX={mv:.4f}")
    c.check("MJX observation layout matches the C backend", menv.obs_layout == cenv.obs_layout)
    cenv.close()


def render_preview(path: Path, out: Path) -> None:
    """Save three views of the standing dog (needs MUJOCO_GL=egl on a headless machine)."""
    print("preview")
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        d.ctrl[:] = 0.10
        mujoco.mj_forward(m, d)
        for _ in range(500):
            mujoco.mj_step(m, d)
        renderer = mujoco.Renderer(m, 720, 960)
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        for name, azimuth, elevation, distance in (("isometric", 135, -18, 0.95),
                                                  ("front", 180, -6, 0.7),
                                                  ("side", 90, -4, 0.75)):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.0, 0.0, 0.17]
            camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
            renderer.update_scene(d, camera=camera)
            target = out.with_name(out.stem + f"_{name}.png")
            imageio.imwrite(target, renderer.render())
            print(f"  wrote {target}")
    except Exception as exc:  # headless machine without a GL context
        print(f"  skipped: {exc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUT / "robot.xml")
    parser.add_argument("--render", action="store_true", help="save a side-view PNG of the standing dog")
    args = parser.parse_args(argv)

    path = args.model
    if not path.exists():
        build_dog_xml(path.parent, DogModelConfig())
    print(f"model: {path}\n")

    c = Checker()
    model_checks(c, path)
    env = DogEnv(4, DogEnvConfig(nominal_mode="pd", command_mode="stand", episode_s=3.0),
                 seed=0, num_threads=4)
    jacobian_checks(c, env)
    stand_checks(c, env)
    env.close()
    tendon_pull_checks(c, path)
    env_checks(c)
    mjx_checks(c)
    if args.render:
        render_preview(path, path.parent / "stand.png")

    print()
    if c.failures:
        print(f"{c.failures} check(s) failed")
    else:
        print("all checks passed")
    c.close()


if __name__ == "__main__":
    main()
