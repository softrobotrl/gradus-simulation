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

## The Onshape leg in Genesis

Everything above is the OpenCR stand-in. The `leg*.py` files bring in the real Gradus leg: the merged `leg.xml` from
Daniel's Onshape export (`mujoco_to_mjcf`, on branch `onshape-to-mjcf-pr`, not on `main` yet). They only read that
file, so Onshape export stays in one place and a rebuilt `leg.xml` is picked up automatically. There is no RL task
for this leg yet; it stops at a model that loads, moves when driven and matches MuJoCo.

```bash
python -m unittest tests.test_genesis_leg_model                    # no Genesis needed
python -m genesis_leg.leg_validate                                 # the synthetic fixture
python -m genesis_leg.leg_validate --xml urdf/gradus/sim/leg.xml   # a real export
python -m genesis_leg.leg_play --xml urdf/gradus/sim/leg.xml       # circle drive in the viewer
```

From Python:

```python
from genesis_leg.leg import GenesisLeg, init_genesis

init_genesis("auto")
leg = GenesisLeg("urdf/gradus/sim/leg.xml", num_envs=64)
leg.step(leg.pull_to_ctrl(pull))   # pull: (num_envs, tendons) in [0, 1]; 0 = rest length, 1 = full stroke
leg.tip(), leg.tension, leg.length
```

### How each part is modelled

| Part | Where it comes from | In Genesis |
|---|---|---|
| Links, mass, inertia | Merged `leg.xml` | Imported from MJCF |
| Joint positions | Calculated in `mujoco_to_mjcf/assembly.py`: on the leg axis, halfway between stations | Imported from MJCF |
| Flexible backbone | Lumped by `merge.py` into ball joints with stiffness `EI / pitch` | Imported: native ball joints and joint stiffness |
| Tendons | Sites and spatial tendons in `leg.xml` | Not imported. `leg_model.py` reads the routes; `leg_tendons.py` computes the forces |
| Servos | `<position>` actuators on tendon length | `T = clamp(kp (L - ctrl) + kv dL/dt, 0, Fmax)`, the same law, applied at each physics step |
| Collision | Mesh geoms | Turned off: nothing to touch yet, and mesh collision is expensive |

Genesis's rigid solver has no flexible-rod element, so the backbone is not simulated natively. Its FEM solver is
volumetric and separate, with no route from this model. The lumped joints are the same approximation MuJoCo uses.

`leg_model.py` writes `leg_genesis.xml` and `leg_routes.json` next to the source file, so mesh paths still resolve.
Only `<position>` servos are supported; a `--motors` export raises an error.

### Results

`leg_validate` runs the same `leg.xml` in both simulators. On Daniel's real export (`Leg-First-Prototype-Print`, built
2026-11-04), all checks pass with Genesis 1.4.3 on CPU:

| Check | Limit | Result |
|---|---|---|
| Mass, degrees of freedom | 0.1%, exact | 1236.2 g, 39 dofs, 52 qs in both |
| Settled foot under sideways gravity | 1 mm | 42.34 mm of droop, 0.057 mm difference |
| Settled foot, each tendon at 25/50/100% stroke | 1 mm or 5% | 53.5 mm of motion at full pull, at most 0.05 mm difference |
| Settled tension at the same pulls | 2 N or 5% | Within 0.1 N at about 142 N |
| Foot path, 6 s circle drive | 2 mm RMS or 5% | 115 mm path, 0.40 mm RMS difference |

All checks also pass on the RTX 4060, with slightly larger settled differences (at most about 0.24 mm in the pulls
sampled) and 0.38 mm RMS on the drive. The model has no `tip` site, so the foot position is the centre of the last
station's tendon holes.

On the synthetic fixture, Genesis 1.4.3, RTX 4060:

| Check | Limit | Result |
|---|---|---|
| Mass, degrees of freedom | 0.1%, exact | 1240 g, 39 dofs, 52 qs in both |
| Settled foot under sideways gravity | 1 mm | 36.73 mm of droop, 0.020 mm difference |
| Settled foot, each tendon at 25/50/100% stroke | 1 mm or 5% | 47 mm of motion at full pull, at most 0.10 mm difference |
| Settled tension at the same pulls | 2 N or 5% | Equal to 0.1 N at about 143 N |
| Foot path, 6 s circle drive | 2 mm RMS or 5% | 109 mm path, 0.39 mm RMS difference |

It also passed on CPU with the leg flipped tip-up and each disk's centre of mass off-centre, as CAD links will be:
about 0.05 mm settled difference at full pull and 0.43 mm RMS on the drive.

Unit tests also compare the force calculation directly with MuJoCo in random bent, moving poses: tendon length, rate
and tension, and the joint torques from the node forces (`qfrc_actuator`), all to 1e-6.

**What this doesn't show:** agreement with MuJoCo isn't agreement with hardware. Both inherit the same joint
placement, lumped rod, inextensible tendons and assumed servo gains. The Onshape export step itself wasn't run here;
the model came from Daniel. The viewer path of `leg_play` hasn't been opened; only `--no-viewer` has run.

### Physics steps

`GenesisLeg` runs 4 physics steps per model timestep (`DEFAULT_SUBSTEPS`, 0.5 ms for a 2 ms model). At 2 ms the
settled pose still matches MuJoCo, but after a full pull the leg swings about twice as long. The foot was 5.4 mm off
MuJoCo 2 s after the pull. At 0.5 ms the motion follows MuJoCo closely; MuJoCo's own result barely changes with
timestep. The cost is proportional. `--substeps` overrides it.

The leg takes seconds to stop swinging after a step in tendon length, so `leg_validate` waits 6 s before comparing
poses. Throughput with these settings (model steps per second, summed over envs):

| Backend | Envs | Model steps/s | Speed vs real time |
|---|---|---|---|
| CPU | 1 | 270 | 0.54x |
| RTX 4060 | 1 | 102 | 0.20x |
| RTX 4060 | 1024 | 80,800 | 0.16x |
| RTX 4060 | 4096 | 147,000 | 0.07x |

The viewer won't reach real time. Use the CPU backend for one env.

### Known limits

- **Ball-joint limits:** Genesis ignores them on import. The merged model doesn't currently set any.
- **Ball-joint stiffness:** Genesis applies it to XYZ Euler angles; MuJoCo uses the rotation vector. They match at
  the small per-joint bends tested here (about 1°) and would diverge at large ones.
- **Windows:** in the conda `gradus-rl` environment, `np.linalg` crashed once torch was imported, which stops the
  mesh loading in `pipeline/` and `leg_fixture.py`. The tests here used a venv with the pip numpy wheel. If you hit a
  silent exit or `0xc06d007f`, replace conda's numpy with pip's.

### Files

| File | Purpose |
|---|---|
| `leg_model.py` | Merged `leg.xml` to `leg_genesis.xml` and tendon routes |
| `leg_tendons.py` | Servo tension and node forces for routes of different lengths |
| `leg.py` | `GenesisLeg`: build the scene, apply tendon forces, read state |
| `leg_fixture.py` | Write a synthetic `leg.xml` in the same format, optionally with your own disk STL (`--stl`) |
| `leg_validate.py` | Compare Genesis against MuJoCo on one `leg.xml` |
| `leg_play.py` | Circle drive, in the viewer or headless |
