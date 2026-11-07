"""Compare the leg in Genesis against MuJoCo, on the same merged leg.xml.

Checks, each printed as PASS/FAIL:
  structure      same mass, joints and degrees of freedom after import
  rest lengths   the routed tendon lengths agree in the starting pose
  droop          with gravity turned sideways, the foot settles in the same place (hanging straight proves nothing)
  pull           each tendon at 25/50/100% stroke: same settled foot position and tension
  circle drive   the same time-varying drive gives the same foot path, and nothing blows up

The limits are the constants below. Agreement with MuJoCo says the Genesis model is a faithful translation, not that
either one matches the hardware.

Usage:
  python -m genesis_leg.leg_validate                                # the synthetic fixture
  python -m genesis_leg.leg_validate --xml urdf/gradus/sim/leg.xml  # a real merged model
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from genesis_leg.leg_play import circle_pull, default_xml

FRACTIONS = (0.25, 0.5, 1.0)
SETTLE_S = 6.0               # the leg swings for seconds after a step; comparing sooner measures ringing, not the pose
DRIVE_S = 6.0
DRIVE_AMPLITUDE, DRIVE_HZ = 0.5, 0.35
MASS_TOLERANCE = 1e-3        # fraction of the total mass
LENGTH_TOLERANCE = 1e-5      # m
DROOP_TOLERANCE = 1e-3       # m
TIP_TOLERANCE = (1e-3, 0.05)  # m, or this fraction of the commanded motion, whichever is larger
TENSION_TOLERANCE = (2.0, 0.05)  # N, or this fraction of the tension, whichever is larger
DRIVE_TOLERANCE = (2e-3, 0.05)   # m RMS, or this fraction of the path's size, whichever is larger


class MujocoLeg:
    """The original model, with the same pull interface as GenesisLeg."""

    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.tip_site = None
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip") >= 0:
            self.tip_site = self.model.site("tip").id

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self.model.actuator_ctrlrange[:, 1]
        mujoco.mj_forward(self.model, self.data)

    def set_pull(self, pull):
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = high - np.clip(pull, 0, 1) * (high - low)

    def step(self, n=1):
        mujoco.mj_step(self.model, self.data, nstep=n)

    def tip(self, body, local):
        if self.tip_site is not None:
            return self.data.site_xpos[self.tip_site].copy()
        b = self.model.body(body).id
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ np.asarray(local)

    def tension(self):
        return -self.data.actuator_force.copy()


def within(error, size, limits):
    absolute, fraction = limits
    return error <= max(absolute, fraction * size)


def pull_cases(n_tendons):
    """One rest case, then each tendon alone at each stroke fraction. Returns (n_cases, n_tendons)."""
    cases = [np.zeros(n_tendons)]
    for tendon in range(n_tendons):
        for fraction in FRACTIONS:
            pull = np.zeros(n_tendons)
            pull[tendon] = fraction
            cases.append(pull)
    return np.array(cases)


def settle_mujoco(mj, cases, steps, tip):
    tips, tensions = [], []
    for pull in cases:
        mj.reset()
        mj.set_pull(pull)
        mj.step(steps)
        if not np.isfinite(mj.data.qpos).all():
            raise SystemExit("MuJoCo itself went unstable on this model")
        tips.append(mj.tip(*tip))
        tensions.append(mj.tension())
    return np.array(tips), np.array(tensions)


def drive_mujoco(mj, names, steps, dt, tip):
    mj.reset()
    path = []
    for i in range(steps):
        mj.set_pull(np.array(circle_pull(names, i * dt, DRIVE_AMPLITUDE, DRIVE_HZ)))
        mj.step()
        path.append(mj.tip(*tip))
    return np.array(path)


def write_sideways_gravity(xml):
    """A copy of leg.xml with gravity along +x, so the straight leg actually has to carry its own weight."""
    root = ET.parse(xml).getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("gravity", "9.81 0 0")
    path = Path(xml).with_name(f"{Path(xml).stem}_sideways.xml")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return path


def check_droop(xml, steps, substeps, results):
    from genesis_leg.leg import GenesisLeg

    sideways = write_sideways_gravity(xml)
    leg = GenesisLeg(sideways, num_envs=1, substeps=substeps)
    tip = (leg.routes["tip_body"], leg.routes["tip_local"])
    for _ in range(steps):
        leg.step()
    mj = MujocoLeg(sideways)
    mj.reset()
    start = mj.tip(*tip)
    mj.step(steps)
    droop = float(np.linalg.norm(mj.tip(*tip) - start))
    error = float(np.linalg.norm(leg.tip()[0].cpu().numpy() - mj.tip(*tip)))
    results.append(("droop", error <= DROOP_TOLERANCE,
                    f"MuJoCo foot moved {droop * 1e3:.2f} mm under sideways gravity; difference {error * 1e3:.3f} mm"))


def run_genesis(xml, cases, settle_steps, drive_steps, substeps):
    import torch

    from genesis_leg.leg import GenesisLeg

    # Every pull case runs at once, plus one last env for the circle drive.
    leg = GenesisLeg(xml, num_envs=len(cases) + 1, substeps=substeps)
    lengths0 = leg.length[0].cpu().numpy().copy()
    pull = torch.zeros(leg.num_envs, len(leg.tendon_names), device=leg.device)
    pull[:-1] = leg._tensor(cases)
    drive_path = []
    tips = tensions = None
    start = time.perf_counter()
    for i in range(max(settle_steps, drive_steps)):
        pull[-1] = leg._tensor(circle_pull(leg.tendon_names, i * leg.dt, DRIVE_AMPLITUDE, DRIVE_HZ))
        leg.step(leg.pull_to_ctrl(pull))
        if i < drive_steps:
            drive_path.append(leg.tip()[-1].cpu().numpy().copy())
        if i == settle_steps - 1:
            tips = leg.tip()[:-1].cpu().numpy().copy()
            tensions = leg.tension[:-1].cpu().numpy().copy()
    elapsed = time.perf_counter() - start
    return leg, lengths0, tips, tensions, np.array(drive_path), elapsed


def check(xml, substeps):
    results = []
    mj = MujocoLeg(xml)
    mj.reset()
    names = [mj.model.tendon(mj.model.actuator_trnid[a, 0]).name for a in range(mj.model.nu)]
    cases = pull_cases(len(names))
    dt = float(mj.model.opt.timestep)
    settle_steps, drive_steps = round(SETTLE_S / dt), round(DRIVE_S / dt)

    leg, lengths0, g_tips, g_tensions, g_path, elapsed = run_genesis(xml, cases, settle_steps, drive_steps, substeps)
    tip = (leg.routes["tip_body"], leg.routes["tip_local"])
    m_tips, m_tensions = settle_mujoco(mj, cases, settle_steps, tip)
    m_path = drive_mujoco(mj, names, drive_steps, dt, tip)
    mj.reset()

    mass = sum(link.get_mass() for link in leg.robot.links)
    n_ball = sum(mj.model.jnt_type == mujoco.mjtJoint.mjJNT_BALL)
    ok = (abs(mass - mj.model.body_mass.sum()) <= MASS_TOLERANCE * mj.model.body_mass.sum()
          and leg.robot.n_dofs == mj.model.nv and leg.robot.n_qs == mj.model.nq)
    results.append(("structure", ok, f"mass {mass * 1e3:.1f} g (MuJoCo {mj.model.body_mass.sum() * 1e3:.1f} g), "
                                     f"{leg.robot.n_dofs} dofs, {leg.robot.n_qs} qs, {n_ball} ball joints"))

    error = float(np.abs(lengths0 - mj.data.ten_length).max())
    results.append(("rest lengths", error <= LENGTH_TOLERANCE, f"max difference {error * 1e3:.4f} mm"))

    check_droop(xml, settle_steps, substeps, results)

    case = 1
    for t, name in enumerate(names):
        for fraction in FRACTIONS:
            motion = float(np.linalg.norm(m_tips[case] - m_tips[0]))
            tip_error = float(np.linalg.norm(g_tips[case] - m_tips[case]))
            tension_error = float(abs(g_tensions[case, t] - m_tensions[case, t]))
            ok = (within(tip_error, motion, TIP_TOLERANCE)
                  and within(tension_error, m_tensions[case, t], TENSION_TOLERANCE))
            results.append((f"pull {name} {fraction:.0%}", ok,
                            f"foot moved {motion * 1e3:6.1f} mm, difference {tip_error * 1e3:5.2f} mm; tension "
                            f"{g_tensions[case, t]:6.1f} N (MuJoCo {m_tensions[case, t]:6.1f} N)"))
            case += 1

    finite = bool(np.isfinite(g_path).all())
    size = float(np.linalg.norm(m_path.max(0) - m_path.min(0)))
    rms = float(np.sqrt(((g_path - m_path) ** 2).sum(-1).mean())) if finite else float("inf")
    results.append(("circle drive", finite and within(rms, size, DRIVE_TOLERANCE),
                    f"path size {size * 1e3:.1f} mm, RMS difference {rms * 1e3:.2f} mm over {DRIVE_S:.0f} s"))

    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    rate = leg.num_envs * max(settle_steps, drive_steps) / elapsed
    print(f"Genesis on {leg.device}: {leg.num_envs} envs, dt {dt * 1e3:g} ms x {substeps} substeps, "
          f"{rate:,.0f} env-steps/s")
    return all(ok for _, ok, _ in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, help="merged leg.xml (default: the synthetic fixture)")
    parser.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--substeps", type=int, help="physics steps per model timestep (default: DEFAULT_SUBSTEPS)")
    args = parser.parse_args()
    from genesis_leg.leg import DEFAULT_SUBSTEPS, init_genesis

    init_genesis(args.backend)
    sys.exit(0 if check(args.xml or default_xml(), args.substeps or DEFAULT_SUBSTEPS) else 1)


if __name__ == "__main__":
    main()
