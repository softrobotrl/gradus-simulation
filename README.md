# Gradus RL - Core Simulation & Control

This repository contains the simulation environment and reinforcement learning control policies for the Gradus RL soft quadruped robot. The ultimate objective is sim-to-real transfer for a tendon-driven physical system.

## Architecture & Stack
We use a monorepo structure to keep the physics bindings and agent training coupled.

*   **Physics/Sim:** Genesis
*   **Control/ML:** PyTorch, Soft Actor-Critic (SAC)
*   **Package Manager:** Conda

## Repository Structure
*   `/pipeline`: CAD to sim pipeline, physics engine definitions, and low-level bindings.
*   `/policy`: SAC implementations, reward shaping, and training loops.

## Environment setup

With Conda installed, run the command for your platform from the repository root.

**macOS:**
```bash
conda env create -f environment-macos.yml
```

**Linux / Windows with an NVIDIA GPU (CUDA):**
```bash
conda env create -f environment.yml
```

Then activate the environment:
```bash
conda activate gradus-rl
```
