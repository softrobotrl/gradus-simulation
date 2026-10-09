"""One command from STL to a validated MJX environment: stage 1 (stl_to_xml), preview image, stage 2 (validate_env).

Usage: python -m pipeline.build pipeline/configs/opencr_tdcr_spatial.yaml
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from pipeline.validate_env import check
from pipeline.stl_to_xml import stl_to_xml


def render_preview(xml_path: Path, pull: float = 0.3, settle_s: float = 0.5) -> Path | None:
    """Side-by-side render: at rest | tendon0 pulled. Skipped when no OpenGL context is available."""
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        renderer = mujoco.Renderer(model, 480, 480)
    except Exception as exc:  # headless machine without EGL/OSMesa
        print(f"preview skipped: {exc}")
        return None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.05]
    camera.distance, camera.azimuth, camera.elevation = 0.35, 120, -15
    frames = []
    for ctrl in (np.zeros(model.nu), np.r_[pull, np.full(model.nu - 1, -pull / 2)]):
        data = mujoco.MjData(model)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data, nstep=round(settle_s / model.opt.timestep))
        renderer.update_scene(data, camera)
        frames.append(renderer.render())
    path = xml_path.with_name("preview.png")
    import matplotlib.pyplot as plt

    plt.imsave(path, np.concatenate(frames, axis=1))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    args = parser.parse_args()
    xml_path = stl_to_xml(args.config)
    if preview := render_preview(xml_path):
        print(f"wrote {preview}")
    sys.exit(0 if check(args.config) else 1)
