# Gradus simulation

Monorepo for the Gradus tendon-driven robots.

## Projects

### Onshape → MuJoCo segmented leg (`onshape_to_mjcf/`)

CAD-driven export: Onshape assembly → `onshape-to-robot` → merged MuJoCo MJCF.

Most recent CAD model:
[Leg-First-Prototype-Print](https://cad.onshape.com/documents/e8da7c50681f40644f3812fb/w/71bd24dd8d18a5e53f7ffb25/e/a388868beab8a28867031630)
(edit access to the Onshape document is required).

- Entry: `python -m onshape_to_mjcf.build`
- After CAD changes, rebuild the sim assembly: `python -m onshape_to_mjcf.build --rebuild`
- Docs: [onshape_to_mjcf/PIPELINE.md](onshape_to_mjcf/PIPELINE.md)
- Output: `urdf/gradus/sim/` (gitignored; regenerate on demand)

No Onshape API key? Use the model zip pinned in the **#reinforcement-learning**
Discord channel.

### Genesis single-leg (`genesis_leg/`)

Single-leg board-pushing environment in Genesis, with explicit elastic cable
forces and PPO training/evaluation.

- Docs: [genesis_leg/README.md](genesis_leg/README.md)
- Shared CAD helper: `pipeline/` (`config.py`, `stl_to_xml.py`, OpenCR STLs/YAML)

### Genesis random terrain (`pipeline/run_go2_genesis.py`)

A Unitree Go2 (the URDF bundled with Genesis, a stand-in until the Gradus quadruped model exists) on random terrain, in the Genesis viewer:

```bash
python -m pipeline.run_go2_genesis --terrain mixed --seed 3 --difficulty 0.5 --backend cpu
```

- `--terrain`: `mixed` (default; a random kind per 2 m tile), `rough`, `slope`, `obstacles` or `flat`.
- `--seed`: the same seed gives the same terrain. Leave it out for a random one; the command to repeat the run is printed.
- `--difficulty`: 0 to 1, a maximum: each tile draws its own difficulty between 0 and this value. At 1: bumps up to 8 cm, slopes up to 20°, blocks 12 cm tall.
- `--grid`: N for an N x N grid of tiles (default 4, so 8 m x 8 m). The robot always starts on a flat tile, so `--grid 1` is flat ground.
- `--backend`: `gpu` (default, NVIDIA only) or `cpu` (use this on macOS).

Go2 holds its standing pose; it has no walking policy yet. Left-drag moves it: grab the side of the body and lift it clear before moving it, because dragging it along the ground tips it over. Rerun the command to reset. The first build takes about 40 s while Genesis compiles; its "SDF pre-processing" and "not watertight" warnings are expected.

To add the terrain to another Genesis scene, use `generate()` from `pipeline/terrain.py` and `add_terrain()` from `pipeline/run_go2_genesis.py`; its docstring lists the limits for batched envs.

## Repository structure

* `onshape_to_mjcf/` — MuJoCo Onshape → MJCF pipeline.
* `genesis_leg/` — Genesis single-leg env + PPO.
* `pipeline/` — shared STL → XML helper and OpenCR assets used by `genesis_leg`; random terrain generator.
* `urdf/gradus/sim/` — generated MuJoCo model (gitignored).
* `urdf/plane/` — ground-plane assets.
* `onshape-mcp/` — Onshape MCP server (git submodule).
* `tests/` — Genesis cable/workflow/terrain checks.
* `paper.pdf` — background paper.

## Quick start

MuJoCo:

```bash
python -m onshape_to_mjcf.build
DISPLAY=:0 python -m mujoco.viewer --mjcf urdf/gradus/sim/scene.xml
```

Genesis:

```bash
python -m unittest discover -s tests -v
python -m genesis_leg.validate --num-envs 4
python -m genesis_leg.train --run-name leg_smoke --num-envs 16 --iterations 2
```

## Environment setup

With Conda installed, run the command for your platform from the repository root.

**macOS:**
```bash
conda env create -f environment-macos.yml
```

**Linux / WSL2 with an NVIDIA GPU (CUDA):**
```bash
conda env create -f environment.yml
```

**Native Windows:**
```bash
conda env create -f environment-windows.yml
```

Then activate the environment:

```bash
conda activate gradus-rl
```

Copy `.env.example` to `.env` and fill in your Onshape API credentials (needed
for `onshape_to_mjcf.build`):

```bash
cp .env.example .env
```
