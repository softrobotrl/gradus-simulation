# Single-leg training in Genesis

Train the OpenCR stand-in leg to push a board along +x. The leg comes from the
STLs and YAML in `pipeline/`; PPO uses `rsl-rl`, as in the Go2 examples.

Run commands from the repository root. For NVIDIA training, use Linux or WSL2.

## Setup

```bash
conda env create -f environment.yml
conda activate gradus-rl
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For an existing environment, use `conda env update -f environment.yml`. On macOS,
use `environment-macos.yml` and pass `--backend cpu` to the commands below. The
CUDA check above is for the NVIDIA machine.

Run `python -m pip check` after setup. The tested combination is PyTorch 2.8.0,
TensorDict 0.10.0, Genesis 1.4.3, and rsl-rl 5.5.1. Update the environment if you
installed the earlier TensorDict 0.15 pin; it requires a newer PyTorch.

Dependencies are pinned in the Conda files. Commands default to `--backend gpu`;
use `--backend cpu` for small local runs. The first run may take longer while
Genesis compiles its kernels.

## Check the environment

```bash
python -m unittest discover -s tests -v
python -m genesis_leg.validate --num-envs 4
python -m genesis_leg.evaluate --baseline zero --episodes 8
python -m genesis_leg.evaluate --baseline scripted --episodes 8 --min-speed-mm-s 1
```

`validate` checks tendon direction, finite observations/rewards, and independent
resets. The scripted baseline checks that the board can be pushed before training.
It sweeps the tip using a 0.6 rad drum amplitude at 0.5 Hz; adjust these with
`--gait-amplitude` and `--gait-hz`. The speed check fails below 1 mm/s or if an
episode ends in a physical failure. Also inspect sideways travel in the results.

Local CPU checks passed: 11 tests, actuation of all three tendons, a 30-iteration
training run, and five more iterations resumed from its checkpoint. The scripted
gait averaged 13.8 mm/s over four 8-second episodes, with 0.32 mm sideways travel
and no failures. GPU execution and the viewer still need checking.

## Train

Start with a short run:

```bash
python -m genesis_leg.train --run-name leg_smoke --num-envs 16 --iterations 2
```

For a CPU smoke run, add `--backend cpu --num-envs 2 --steps-per-env 8`.

Then start a longer GPU run:

```bash
python -m genesis_leg.train --run-name single_leg_v1 --num-envs 256 --iterations 1000
tensorboard --logdir logs/genesis_leg
```

An iteration collects 24 steps per environment by default. Check throughput and
memory before increasing the batch size. The number of iterations needed for a
useful policy hasn't been established.

Runs go in `logs/genesis_leg/<run-name>/`, with `config.json`, TensorBoard events,
and periodic `model_<iteration>.pt` checkpoints. Every checkpoint also refreshes
`latest.pt`; saves use a temporary file and atomic replacement. The default save
interval is 50 iterations (`--save-interval`), with another save at completion or
Ctrl+C. Existing run directories are kept; choose a new name for a new run.

```bash
python -m genesis_leg.train --resume logs/genesis_leg/single_leg_v1/latest.pt --iterations 1000
```

Resume restores PPO weights, optimizer, and observation normalization. It starts
fresh simulation episodes and random streams. `--iterations` means additional
iterations. You can change `--num-envs` and `--seed`; each resume records its
settings in `resume_<timestamp>.json`.

New runs also accept `--config`, `--cable-stiffness`, `--cable-damping`,
`--action-filter`, and `--episode-s`. Resume reuses the saved physics settings.

## Evaluate and view

```bash
python -m genesis_leg.evaluate logs/genesis_leg/single_leg_v1/latest.pt --episodes 32 --out results/single_leg_v1/evaluation.json
python -m genesis_leg.evaluate logs/genesis_leg/single_leg_v1/latest.pt --backend cpu --view
```

Evaluation runs the deterministic policy for one episode per environment. It
reports forward speed, sideways travel, return, failures, and peak cable tension.
`--view` opens a single episode in the Genesis viewer and requires a display.
Use the name of a run you have actually trained: `single_leg_v1` above is an
example. To evaluate on another machine, copy the checkpoint and its `config.json`
together, and use the same robot YAML and assets from the repository.

## Model assumptions

`model.py` reuses the CAD pipeline to write
`pipeline/build/genesis/scene-*/<name>/board_scene.xml` and cable routes. Each
environment build gets a separate directory so training and evaluation can start
together. MuJoCo prepares the geometry and resting cable lengths; Genesis handles
the simulation.

Genesis doesn't import this model's spatial tendons, so `cables.py` calculates
spring/damper forces along the carriage, pulley, and disk-hole path. `env.py`
applies those forces and their moments at each physics step. The servo carriages
keep their reflected motor inertia, and drum PD torque is limited by the YAML.

Cable stiffness (2000 N/m) and damping (1 N*s/m) are assumptions that need
calibration. The model treats guides as frictionless and cables as elastic; it
doesn't reproduce MuJoCo's tendon-length constraints. Cable tension can exceed
the motor's stall force during motion because the limit applies to motor torque.
Genesis doesn't enforce the ball-joint cone limits, so exceeding the YAML angle
ends the episode. Contact and rail behavior also need comparison with hardware.

The policy has three actions in [-1, 1]: negative releases a tendon, zero is taut,
and positive winds it. Observations contain joint positions and velocities, servo
targets, normalized tensions, and tip position (98 values for this model). Board
position is included so the policy can see its edges.

Reward is forward speed times 100, minus sideways speed times 50, effort, and
command-change costs. Time limits and board-edge exits are truncations; excessive
bend or invalid observations are failures. Fatal solver errors stop the run;
the last complete checkpoint can be resumed. The environment returns the final
observation separately from the observation after an automatic reset. `LegPPO`
uses that final observation for timeout value estimates; physical failures don't
bootstrap future rewards.

## Files

| File | Purpose |
|---|---|
| `model.py` | Build the scene and extract cable routes |
| `cables.py` | Calculate cable tensions and guide forces |
| `env.py` | Step physics, calculate rewards, and reset episodes |
| `validate.py` | Check actuation and resets |
| `train.py` | Train PPO and save or resume runs |
| `runner.py` | Bootstrap timeouts from final observations and save checkpoints atomically |
| `evaluate.py` | Run a saved policy or baseline, with an optional viewer |
