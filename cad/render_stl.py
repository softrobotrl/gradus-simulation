"""Render the exported dog STL to PNG views, without Onshape.

Onshape's own PNG export needs the translation service to be healthy.  This
script renders the same mesh locally with MuJoCo's renderer, so the geometry
can always be looked at.  Usage::

    python -m cad.render_stl                              # uses the exported STL
    python -m cad.render_stl --stl path/to/other.stl --out /tmp/views

Needs an OpenGL context: set MUJOCO_GL=egl on a headless machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_STL = Path("/home/daniel/Downloads/gradus_dog/gradus_dog.stl")
DEFAULT_OUT = Path("/home/daniel/Downloads/gradus_dog")

# name, azimuth, elevation, distance
VIEWS = (
    ("isometric", 135.0, -24.0, 0.88),
    ("front", 180.0, -6.0, 0.62),
    ("side", 90.0, -4.0, 0.68),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    import mujoco
    import imageio.v2 as imageio
    import trimesh

    mesh = trimesh.load(args.stl, force="mesh")
    # MuJoCo's STL reader wants a binary STL; Onshape writes ASCII.
    binary = args.out / "gradus_dog_binary.stl"
    binary.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(binary)
    print(f"mesh: {len(mesh.faces)} faces, bbox {np.round(mesh.bounds, 3).tolist()}")

    xml = f"""
    <mujoco>
      <visual><global offwidth="{args.width}" offheight="{args.height}"/></visual>
      <asset>
        <mesh name="dog" file="{binary}"/>
        <material name="shell" rgba="0.82 0.83 0.85 1"/>
      </asset>
      <worldbody>
        <geom name="ground" type="plane" size="1.2 1.2 0.1" rgba="0.97 0.97 0.97 1"/>
        <body name="dog">
          <geom name="mesh" type="mesh" mesh="dog" material="shell" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, args.height, args.width)
    centre = (mesh.bounds[0] + mesh.bounds[1]) / 2
    for name, azimuth, elevation, distance in VIEWS:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [centre[0], centre[1], centre[2] * 0.9]
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera)
        path = args.out / f"gradus_dog_{name}.png"
        imageio.imwrite(path, renderer.render())
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
