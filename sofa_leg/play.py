"""Drive the SOFA leg's tendons so the foot traces a circle, headless (or in a window with --gui).

Same tension-only, phase-shifted drive as the MuJoCo and Genesis players (genesis_leg.leg_play.circle_pull).

Usage:
  python -m sofa_leg.play --seconds 2
  python -m sofa_leg.play --gui                  # SOFA's own viewer; or use runSofa with sofa_leg/run_scene.py
  python -m sofa_leg.play --seconds 6 --csv tip.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np

from genesis_leg.leg_play import circle_pull
from sofa_leg.leg import DEFAULT_DT, SofaLeg
from sofa_leg.spec import GRIPS, RodParams

DEFAULT_XML = Path(__file__).resolve().parents[1] / "urdf" / "gradus" / "sim" / "leg.xml"


def gui(xml):
    """Open the leg in SOFA's ImGui viewer, driven by sofa_leg.run_scene's controller."""
    import os

    import Sofa.Core
    import Sofa.Gui
    import Sofa.Simulation
    import SofaRuntime

    from sofa_leg import run_scene

    SofaRuntime.importPlugin("SofaImGui")
    os.environ.setdefault("SOFA_LEG_XML", str(xml))
    root = Sofa.Core.Node("root")
    run_scene.createScene(root)
    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("sofa_leg", "imgui")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1280, 900)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--amplitude", type=float, default=0.5, help="peak pull as a fraction of the servo stroke")
    parser.add_argument("--hz", type=float, default=0.35)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--grip", choices=GRIPS, default="thickness")
    parser.add_argument("--csv", type=Path, help="write time, tip x y z and the 8 tensions")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.gui:
        return gui(args.xml)

    leg = SofaLeg(args.xml, rod=RodParams(grip=args.grip), dt=args.dt)
    leg.reset()
    rows, start = [], time.perf_counter()
    for i in range(round(args.seconds / leg.dt)):
        t = i * leg.dt
        leg.set_pull(np.array(circle_pull(leg.tendon_names, t, args.amplitude, args.hz)))
        leg.step()
        rows.append(np.concatenate([[t + leg.dt], leg.tip(), leg.tension()]))
    rows = np.array(rows)
    if not np.isfinite(rows).all():
        raise SystemExit("FAIL: the simulation produced non-finite values")
    elapsed = time.perf_counter() - start
    span = (rows[:, 1:4].max(0) - rows[:, 1:4].min(0)) * 1000
    print(f"{args.seconds:.1f} s simulated in {elapsed:.1f} s ({args.seconds / elapsed:.3f}x real time, "
          f"{len(rows) / elapsed:.1f} steps/s); foot swept {span[0]:.1f} x {span[1]:.1f} x {span[2]:.1f} mm; "
          f"peak tension {rows[:, 4:].max():.1f} N")
    if args.csv:
        header = "t,tip_x,tip_y,tip_z," + ",".join(f"tension_{n.removeprefix('tendon_')}" for n in leg.tendon_names)
        np.savetxt(args.csv, rows, delimiter=",", header=header, comments="")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
