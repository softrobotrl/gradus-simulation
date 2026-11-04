# Gradus simulation

Onshape → MuJoCo pipeline for the segmented tendon-driven leg.

## Repository structure

* `cad/` — the Onshape → MJCF pipeline. Entry point: `cad/build_leg_mjcf.py`.
  Full docs: [cad/LEG_MJCF_PIPELINE.md](cad/LEG_MJCF_PIPELINE.md).
* `urdf/gradus/sim/` — generated MuJoCo model (`leg.xml`, `scene.xml`,
  `assets/`). Gitignored; regenerate with the command below.
* `urdf/plane/` — ground-plane assets.
* `onshape-mcp/` — Onshape MCP server (git submodule).
* `paper.pdf` — background paper.

## Quick start

```bash
python -m cad.build_leg_mjcf      # regenerate urdf/gradus/sim/leg.xml
DISPLAY=:0 python -m mujoco.viewer --mjcf urdf/gradus/sim/scene.xml
```

Animated tendon demo:

```bash
DISPLAY=:0 python -m cad.play_leg --xml urdf/gradus/sim/scene.xml
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
