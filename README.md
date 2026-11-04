# Gradus simulation

Monorepo for the Gradus tendon-driven robots.

## Projects

### Genesis single-leg (`genesis_leg/`)

Single-leg board-pushing environment in Genesis, with explicit elastic cable
forces and PPO training/evaluation.

- Docs: [genesis_leg/README.md](genesis_leg/README.md)
- Shared CAD helper: `pipeline/` (`config.py`, `stl_to_xml.py`, OpenCR STLs/YAML)

## Repository structure

* `genesis_leg/` — Genesis single-leg env + PPO.
* `pipeline/` — shared STL → XML helper and OpenCR assets used by `genesis_leg`.
* `urdf/plane/` — ground-plane assets.
* `onshape-mcp/` — Onshape MCP server (git submodule).
* `tests/` — Genesis cable/workflow checks.
* `paper.pdf` — background paper.

## Quick start

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
