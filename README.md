# Gradus RL - Core Simulation & Control

This repository contains the simulation environment and reinforcement learning control policies for the Gradus RL soft quadruped robot. The ultimate objective is sim-to-real transfer for a tendon-driven physical system.

## Architecture & Stack
We use a monorepo structure to keep the physics bindings and agent training coupled.

*   **Physics/Sim:** Genesis
*   **Control/ML:** PyTorch, Soft Actor-Critic (SAC)
*   **Package Manager:** Conda

## Repository Structure
*   `/systems`: Owns the physics baseline, CAD imports, physics engine definitions, and low-level bindings.
*   `/research`: Owns the algorithmic baseline, SAC implementations, reward shaping, and training loops.
