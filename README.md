# Gradus RL - Core Simulation & Control

This repository contains the simulation environment and reinforcement learning control policies for the Gradus RL soft quadruped robot. The ultimate objective is sim-to-real transfer for a tendon-driven physical system.

## Architecture & Stack
We use a monorepo structure to keep the physics bindings and agent training coupled.

*   **Physics/Sim:** MuJoCo MJX
*   **Control/ML:** PyTorch, Soft Actor-Critic (SAC)
*   **Package Manager:** Conda

## Repository Structure
*   `/pipeline`: CAD to sim pipeline, physics engine definitions, and low-level bindings. See [pipeline/README.md](pipeline/README.md) for the STL to MuJoCo MJX pipeline and how to run it.
    *   Single-leg board-pushing task for SAC training: [pipeline/BOARD_TASK.md](pipeline/BOARD_TASK.md).
*   `/policy`: SAC implementations, reward shaping, and training loops. See [policy/README.md](policy/README.md) to train SAC on the board-pushing task (`python -m policy.train_sac`).

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

**Native Windows** (PyTorch on the GPU; the physics runs on CPU threads, because JAX has no CUDA build for Windows):
```bash
conda env create -f environment-windows.yml
```
If conda refuses with a Terms of Service error, see [policy/README.md](policy/README.md#quick-start).

Then activate the environment:
```bash
conda activate gradus-rl
```
