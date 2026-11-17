# Gradus simulation

Monorepo for the Gradus tendon-driven robots.

## Projects

### MuJoCo segmented leg (`mujoco_to_mjcf/`)

Onshape → `onshape-to-robot` → merged MuJoCo MJCF.

- Entry: `python -m mujoco_to_mjcf.build`
- Docs: [mujoco_to_mjcf/PIPELINE.md](mujoco_to_mjcf/PIPELINE.md)
- Output: `urdf/gradus/sim/` (gitignored; regenerate on demand)

### Genesis single-leg (`genesis_leg/`)

Single-leg board-pushing environment in Genesis, with explicit elastic cable
forces and PPO training/evaluation.

- Docs: [genesis_leg/README.md](genesis_leg/README.md)
- Shared CAD helper: `pipeline/` (`config.py`, `stl_to_xml.py`, OpenCR STLs/YAML)

### Demo leg on a sliding rig (`single_leg_demo/`)

The `Sim-Leg-Assembly` demo leg from Onshape, mounted on a one-dimensional
linear rail. PPO drives the eight tendon servos so the carriage tracks a
commanded velocity and pumps itself back and forth.

- Build: `python -m mujoco_to_mjcf.build --out urdf/gradus/demo` then
  `python -m single_leg_demo.build_rig`
- Train: `python -m single_leg_demo.train --num-envs 64 --iterations 1000`
- Evaluate: `python -m single_leg_demo.evaluate --ckpt <checkpoint> --cmd-min 0.10 --cmd-max 0.10`
- Docs: [single_leg_demo/README.md](single_leg_demo/README.md)

The CAD source is the `Demo-Leg` document
(`ee938c509aab09f13ed20423` / `Sim-Leg-Assembly`). Tendon routing is unchanged
from the original leg: 4 inner + 4 outer spatial tendons (8 servos).

## Repository structure

* `mujoco_to_mjcf/` — MuJoCo Onshape → MJCF pipeline.
* `single_leg_demo/` — demo leg sliding-rig env + PPO.
* `walking_box/` — four-leg box quadruped env + PPO.
* `genesis_leg/` — Genesis single-leg env + PPO.
* `pipeline/` — shared STL → XML helper and OpenCR assets used by `genesis_leg`.
* `urdf/gradus/sim/` — generated MuJoCo model (gitignored).
* `urdf/gradus/demo/` — generated demo-leg and rig model (gitignored).
* `urdf/plane/` — ground-plane assets.
* `onshape-mcp/` — Onshape MCP server (git submodule).
* `tests/` — cable, workflow and sliding-rig checks.
* `paper.pdf` — background paper.

## Quick start

MuJoCo:

```bash
python -m mujoco_to_mjcf.build
DISPLAY=:0 python -m mujoco.viewer --mjcf urdf/gradus/sim/scene.xml
```

Genesis:

```bash
python -m unittest discover -s tests -v
python -m genesis_leg.validate --num-envs 4
python -m genesis_leg.train --run-name leg_smoke --num-envs 16 --iterations 2
```

Demo leg on the sliding rig:

```bash
set -a && . ./.env && set +a
python -m mujoco_to_mjcf.build --out urdf/gradus/demo
python -m single_leg_demo.build_rig
python -m single_leg_demo.validate --num-envs 4
python -m single_leg_demo.train --run-name rig_smoke --num-envs 16 --iterations 5
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
