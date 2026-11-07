"""Compare the SOFA leg (FEM rod) with the MuJoCo leg (ball joints + springs) on the same leg.xml.

Unlike genesis_leg.leg_validate, this is not a translation check: the two models are meant to differ (a continuous
FEM rod clamped by the disks vs 13 lumped spring joints with joint friction). So only a few things are gated:

  gated     total disk mass, rest tendon lengths (same geometry), everything finite, and every pull moving the foot
            the same way as in MuJoCo (direction cosine > 0.9, size within x2)
  reported  sideways-gravity droop, each tendon at 25/50/100% stroke (foot position and tension), the 6 s circle
            drive (RMS path difference), each against MuJoCo as-is and against MuJoCo with its joint friction removed

Settled poses use dt = 4 ms and stop early once the foot is still (equilibrium does not depend on dt); the drive
uses the leg's normal dt. Cases run in parallel processes, one SOFA scene each.

Usage:
  python -m sofa_leg.validate                       # urdf/gradus/sim/leg.xml, both grip modes
  python -m sofa_leg.validate --grip thickness --workers 12
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from genesis_leg.leg_play import circle_pull
from genesis_leg.leg_validate import (DRIVE_AMPLITUDE, DRIVE_HZ, DRIVE_S, FRACTIONS, SETTLE_S, MujocoLeg, pull_cases,
                                      write_sideways_gravity)
from sofa_leg.spec import GRIPS, RodParams

REPO = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO / "urdf" / "gradus" / "sim" / "leg.xml"
OUT_DIR = REPO / "pipeline" / "build" / "sofa"
STATIC_DT = 0.004
STILL_SPEED = 1e-5        # m/s: the foot counts as settled below this...
STILL_FOR = 0.2           # ...for this long
MIN_SETTLE = 1.0          # s
LENGTH_TOLERANCE = 1e-6   # m
MASS_TOLERANCE = 1e-6     # kg


# ---- SOFA workers (run in their own processes) -----------------------------------------------------------------
def _sofa_settle(job):
    from sofa_leg.leg import SofaLeg

    leg = SofaLeg(job["xml"], rod=RodParams(**job["rod"]), dt=STATIC_DT, gravity=job.get("gravity"))
    leg.reset()
    leg.set_pull(job["pull"])
    start, still, steps = time.perf_counter(), 0.0, 0
    chunk = max(1, round(0.05 / STATIC_DT))
    while steps * STATIC_DT < SETTLE_S:
        before = leg.tip()
        leg.step(chunk)
        steps += chunk
        if not leg.finite():
            break
        speed = np.linalg.norm(leg.tip() - before) / (chunk * STATIC_DT)
        still = still + chunk * STATIC_DT if speed < STILL_SPEED else 0.0
        if steps * STATIC_DT >= MIN_SETTLE and still >= STILL_FOR:
            break
    return {"tip": leg.tip(), "tension": leg.tension(), "length": leg.length(), "finite": leg.finite(),
            "seconds_simulated": steps * STATIC_DT, "wall": time.perf_counter() - start,
            "rest_length": None, "disk_mass": leg.spec.disk_mass}


def _sofa_rest(job):
    from sofa_leg.leg import SofaLeg

    leg = SofaLeg(job["xml"], rod=RodParams(**job["rod"]))
    leg.reset()
    nodes = leg.rod_nodes()
    rod_mass = RodParams(**job["rod"]).density * np.pi * leg.spec.rod.radius ** 2 * (leg.spec.rod_span[1] - leg.spec.rod_span[0])
    return {"length": leg.length(), "disk_mass": leg.spec.disk_mass, "rod_mass": rod_mass,
            "n_rod_nodes": len(nodes), "n_free_nodes": leg.n_free_nodes, "rayleigh": leg.handle.rayleigh,
            "dt": leg.dt}


def _sofa_drive(job):
    from sofa_leg.leg import SofaLeg

    leg = SofaLeg(job["xml"], rod=RodParams(**job["rod"]))
    leg.reset()
    every = round(job["sample_dt"] / leg.dt)
    path, start = [], time.perf_counter()
    for i in range(round(DRIVE_S / leg.dt)):
        leg.set_pull(np.array(circle_pull(leg.tendon_names, i * leg.dt, DRIVE_AMPLITUDE, DRIVE_HZ)))
        leg.step()
        if (i + 1) % every == 0:
            path.append(leg.tip())
        if not leg.finite():
            break
    return {"path": np.array(path), "wall": time.perf_counter() - start, "steps": round(DRIVE_S / leg.dt)}


# ---- MuJoCo references -------------------------------------------------------------------------------------------
def _mujoco_settle(xml, cases, frictionless):
    mj = MujocoLeg(xml)
    if frictionless:
        mj.model.dof_frictionloss[:] = 0
    steps = round(SETTLE_S / mj.model.opt.timestep)
    tips, tensions = [], []
    for pull in cases:
        mj.reset()
        mj.set_pull(pull)
        mj.step(steps)
        tips.append(mj.tip(None, None) if mj.tip_site is not None else _mj_tip(mj))
        tensions.append(mj.tension())
    return np.array(tips), np.array(tensions)


def _mj_tip(mj):
    from genesis_leg.leg_model import find_tip, read_routes

    body, local = find_tip(mj.model, read_routes(mj.model))
    return mj.tip(body, local)


def _mujoco_drive(xml, frictionless):
    mj = MujocoLeg(xml)
    if frictionless:
        mj.model.dof_frictionloss[:] = 0
    names = [mj.model.tendon(mj.model.actuator_trnid[a, 0]).name for a in range(mj.model.nu)]
    dt = float(mj.model.opt.timestep)
    mj.reset()
    path = []
    for i in range(round(DRIVE_S / dt)):
        mj.set_pull(np.array(circle_pull(names, i * dt, DRIVE_AMPLITUDE, DRIVE_HZ)))
        mj.step()
        path.append(_mj_tip(mj))
    return np.array(path), dt, names


# ---- comparison -----------------------------------------------------------------------------------------------------
def compare(xml, grip, workers):
    rod = asdict(RodParams(grip=grip))
    mj = MujocoLeg(xml)
    mj.reset()
    names = [mj.model.tendon(mj.model.actuator_trnid[a, 0]).name for a in range(mj.model.nu)]
    cases = pull_cases(len(names))
    sideways = write_sideways_gravity(xml)

    print(f"\n=== grip = {grip}: running {len(cases) + 3} SOFA jobs on {workers} processes ...", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        drive_future = pool.submit(_sofa_drive, {"xml": str(xml), "rod": rod, "sample_dt": float(mj.model.opt.timestep)})
        rest_future = pool.submit(_sofa_rest, {"xml": str(xml), "rod": rod})
        droop_future = pool.submit(_sofa_settle, {"xml": str(xml), "rod": rod, "pull": np.zeros(len(names)),
                                                  "gravity": [9.81, 0, 0]})
        settle_futures = [pool.submit(_sofa_settle, {"xml": str(xml), "rod": rod, "pull": pull}) for pull in cases]
        settles = [f.result() for f in settle_futures]
        droop, rest, drive = droop_future.result(), rest_future.result(), drive_future.result()
    wall = time.perf_counter() - start

    references = {}
    for frictionless in (False, True):
        tips, tensions = _mujoco_settle(xml, cases, frictionless)
        side_tips, _ = _mujoco_settle(sideways, cases[:1], frictionless)
        straight, _ = _mujoco_settle(xml, cases[:1], frictionless)
        path, mj_dt, _ = _mujoco_drive(xml, frictionless)
        references[frictionless] = (tips, tensions, side_tips[0], straight[0], path)

    results, report = [], {"grip": grip, "rod": rod, "wall_s": wall, "rest": {k: v for k, v in rest.items()}}
    s_tips = np.array([s["tip"] for s in settles])
    s_tensions = np.array([s["tension"] for s in settles])
    finite = all(s["finite"] for s in settles) and droop["finite"] and np.isfinite(drive["path"]).all()

    mass_error = abs(rest["disk_mass"] - mj.model.body_mass.sum())
    results.append(("disk mass", mass_error <= MASS_TOLERANCE,
                    f"{rest['disk_mass'] * 1e3:.1f} g (MuJoCo {mj.model.body_mass.sum() * 1e3:.1f} g); "
                    f"plus the rod {rest['rod_mass'] * 1e3:.1f} g, which MuJoCo leaves out"))
    length_error = float(np.abs(rest["length"] - mj.data.ten_length).max())
    results.append(("rest lengths", length_error <= LENGTH_TOLERANCE, f"max difference {length_error * 1e3:.2e} mm"))
    results.append(("finite", finite, "every settle, the droop and the drive"))

    tips, tensions, side_tip, straight, path = references[False]
    tips_nf, tensions_nf, side_tip_nf, straight_nf, path_nf = references[True]
    droop_mj = float(np.linalg.norm(side_tip - straight))
    droop_sofa = float(np.linalg.norm(droop["tip"] - s_tips[0]))
    droop_nf = float(np.linalg.norm(side_tip_nf - straight_nf))
    report["droop"] = {"sofa_mm": droop_sofa * 1e3, "mujoco_mm": droop_mj * 1e3, "mujoco_frictionless_mm": droop_nf * 1e3}
    results.append(("droop", None, f"sideways gravity moves the foot {droop_sofa * 1e3:.2f} mm in SOFA, "
                                   f"{droop_mj * 1e3:.2f} mm in MuJoCo ({droop_nf * 1e3:.2f} mm without joint friction)"))

    report["pulls"] = []
    case = 1
    for t, name in enumerate(names):
        for fraction in FRACTIONS:
            m_move = tips[case] - tips[0]
            s_move = s_tips[case] - s_tips[0]
            nf_move = tips_nf[case] - tips_nf[0]
            size = np.linalg.norm(m_move)
            cosine = float(m_move @ s_move / max(size * np.linalg.norm(s_move), 1e-12))
            ratio = float(np.linalg.norm(s_move) / max(size, 1e-12))
            ok = size < 1e-3 or (cosine > 0.9 and 0.5 <= ratio <= 2.0)
            row = {"tendon": name, "fraction": fraction, "sofa_move_mm": np.linalg.norm(s_move) * 1e3,
                   "mujoco_move_mm": size * 1e3, "mujoco_frictionless_move_mm": np.linalg.norm(nf_move) * 1e3,
                   "difference_mm": float(np.linalg.norm(s_tips[case] - tips[case]) * 1e3),
                   "difference_frictionless_mm": float(np.linalg.norm(s_tips[case] - tips_nf[case]) * 1e3),
                   "cosine": cosine, "sofa_tension": float(s_tensions[case, t]),
                   "mujoco_tension": float(tensions[case, t]), "settle_s": settles[case]["seconds_simulated"]}
            report["pulls"].append(row)
            results.append((f"pull {name} {fraction:.0%}", ok,
                            f"foot moves {row['sofa_move_mm']:6.1f} mm (MuJoCo {row['mujoco_move_mm']:6.1f}, "
                            f"no friction {row['mujoco_frictionless_move_mm']:6.1f}), cos {cosine:5.3f}; "
                            f"tension {row['sofa_tension']:5.1f} N (MuJoCo {row['mujoco_tension']:5.1f})"))
            case += 1

    n = min(len(drive["path"]), len(path))
    size = float(np.linalg.norm(path.max(0) - path.min(0)))
    rms = float(np.sqrt(((drive["path"][:n] - path[:n]) ** 2).sum(-1).mean()))
    rms_nf = float(np.sqrt(((drive["path"][:n] - path_nf[:n]) ** 2).sum(-1).mean()))
    s_size = float(np.linalg.norm(drive["path"].max(0) - drive["path"].min(0)))
    report["drive"] = {"sofa_size_mm": s_size * 1e3, "mujoco_size_mm": size * 1e3, "rms_mm": rms * 1e3,
                       "rms_frictionless_mm": rms_nf * 1e3, "wall_s": drive["wall"],
                       "steps_per_s": drive["steps"] / drive["wall"]}
    results.append(("circle drive", None, f"path size {s_size * 1e3:.1f} mm (MuJoCo {size * 1e3:.1f} mm); RMS difference "
                                          f"{rms * 1e3:.2f} mm ({rms_nf * 1e3:.2f} mm vs frictionless MuJoCo) "
                                          f"over {DRIVE_S:.0f} s"))

    width = max(len(r[0]) for r in results)
    for label, ok, detail in results:
        print(f"{'INFO' if ok is None else 'PASS' if ok else 'FAIL'}  {label:<{width}}  {detail}")
    print(f"SOFA: {rest['n_rod_nodes']} rod nodes ({rest['n_free_nodes']} free), dt {rest['dt'] * 1e3:g} ms, Rayleigh "
          f"beta {rest['rayleigh']:.4g} s; drive ran at {report['drive']['steps_per_s']:.1f} steps/s "
          f"({report['drive']['steps_per_s'] * rest['dt']:.3f}x real time); all jobs {wall:.0f} s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"validate_{grip}.json"
    out.write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    print(f"wrote {out}")
    return all(ok for _, ok, _ in results if ok is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--grip", choices=GRIPS + ("both",), default="both")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    grips = GRIPS if args.grip == "both" else (args.grip,)
    ok = all([compare(args.xml.resolve(), grip, args.workers) for grip in grips])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
