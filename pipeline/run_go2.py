"""Unitree Go2 standing on random terrain in the MuJoCo viewer.

Usage: python -m pipeline.run_go2 --terrain mixed --seed 3 --difficulty 0.5 --grid 4

Go2 starts on a flat tile and holds its home pose with its built-in position servos (there is no policy yet); push it
with Ctrl + right-drag.
Backspace resets to the model's default pose at the origin, not the spawn point: rerun the command instead.

Go2 is a stand-in until the Gradus quadruped model exists. On first run its MJX model (~30 MB) is downloaded from
mujoco_menagerie (BSD-3, see pipeline/assets/unitree_go2/LICENSE) at a pinned commit into pipeline/assets/unitree_go2/,
which is gitignored.
"""

import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from pipeline.terrain import Terrain, add_terrain

# Files are only downloaded when missing: after changing the commit, delete pipeline/assets/unitree_go2/.
MENAGERIE = ("https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
             "6da8b971844845de6ab27315aacc3a3fea4ddb8e/unitree_go2")
GO2_DIR = Path(__file__).parent / "assets" / "unitree_go2"


def _download(name: str) -> Path:
    dest = GO2_DIR / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading unitree_go2/{name}")
        part = dest.with_name(dest.name + ".part")  # an interrupted download never leaves a half file at dest
        urllib.request.urlretrieve(f"{MENAGERIE}/{name}", part)
        part.rename(dest)
    return dest


def fetch_go2() -> Path:
    """Download the MJX version of Go2 once; return its scene file."""
    _download("LICENSE")
    for mesh in ET.parse(_download("go2_mjx.xml")).iter("mesh"):
        _download(f"assets/{mesh.get('file')}")
    return _download("scene_mjx.xml")


def build_model(terrain: Terrain) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(fetch_go2()))
    spec.delete(spec.geom("floor"))  # the terrain is the floor; MuJoCo cannot collide a plane with a height field
    for geom in spec.geoms:
        geom.margin = 0.0  # menagerie uses 0.001; MJX cannot collide a height field with a nonzero margin
    add_terrain(spec, terrain, material="groundplane")
    return spec.compile()


def spawn(model: mujoco.MjModel, terrain: Terrain) -> mujoco.MjData:
    """Go2 in its home pose at the spawn tile, raised so its feet just clear the highest terrain under it."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    x, y = terrain.spawn_xy
    data.qpos[:3] = x, y, terrain.max_height(x, y) + data.qpos[2] + 0.02
    mujoco.mj_forward(model, data)
    return data
