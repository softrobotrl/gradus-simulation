"""Drive the leg's tendons so the foot traces a circle, in the Genesis viewer or headless.

Same tension-only, phase-shifted drive as mujoco_to_mjcf/play.py.

Usage:
  python -m genesis_leg.leg_play --xml urdf/gradus/sim/leg.xml
  python -m genesis_leg.leg_play --no-viewer --seconds 6       # no --xml: the synthetic fixture
"""

import argparse
import math
import time
from pathlib import Path

import torch

OUTER_SCALE, OUTER_PHASE = 0.45, 0.6


def circle_pull(names, t, amplitude, hz):
    """Pull fraction in [0, 1] per tendon at time t; each direction's tendon pulls for half of the circle."""
    w = 2 * math.pi * hz * t
    pull = []
    for name in names:
        _, kind, direction = name.split("_")
        scale, angle = (OUTER_SCALE, w + OUTER_PHASE) if kind == "outer" else (1.0, w)
        wave = {"front": math.cos(angle), "back": -math.cos(angle), "left": math.sin(angle), "right": -math.sin(angle)}
        pull.append(amplitude * scale * max(0.0, wave[direction]))
    return pull


def default_xml():
    from genesis_leg.leg_fixture import BUILD_ROOT, write_fixture

    return write_fixture(BUILD_ROOT / "fixture_leg")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, help="merged leg.xml (default: the synthetic fixture)")
    parser.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--amplitude", type=float, default=0.5, help="peak pull as a fraction of the servo stroke")
    parser.add_argument("--hz", type=float, default=0.35)
    parser.add_argument("--substeps", type=int, help="physics steps per model timestep (default: DEFAULT_SUBSTEPS)")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    from genesis_leg.leg import DEFAULT_SUBSTEPS, GenesisLeg, init_genesis

    init_genesis(args.backend)
    leg = GenesisLeg(args.xml or default_xml(), substeps=args.substeps or DEFAULT_SUBSTEPS, show_viewer=not args.no_viewer)
    tips = []
    start = time.perf_counter()
    for i in range(round(args.seconds / leg.dt)):
        t = i * leg.dt
        pull = leg._tensor(circle_pull(leg.tendon_names, t, args.amplitude, args.hz))
        leg.step(leg.pull_to_ctrl(pull)[None])
        tips.append(leg.tip()[0].clone())
        if not args.no_viewer:
            time.sleep(max(0.0, start + t - time.perf_counter()))  # real time
    tips = torch.stack(tips)
    if not torch.isfinite(tips).all():
        raise SystemExit("FAIL: the simulation produced non-finite values")
    span = (tips.max(0).values - tips.min(0).values) * 1000
    print(f"{args.seconds:.1f} s simulated in {time.perf_counter() - start:.1f} s; foot swept "
          f"{span[0]:.1f} x {span[1]:.1f} x {span[2]:.1f} mm; peak tension {leg.tension.max():.1f} N at the end")


if __name__ == "__main__":
    main()
