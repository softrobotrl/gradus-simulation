# STL to MJX pipeline

Turns the hardware team's STL parts into a simulated robot whose tendons are pulled by servos. It runs in MJX, the version of MuJoCo that runs thousands of simulations in parallel on a GPU. The objective is to have a trainable environment within 30 minutes of receiving STL files from the hardware team.

The pipeline has two stages: STL to XML (builds the MuJoCo robot description) and XML to sim env (wraps it in a trainable environment and checks it works). `build.py` runs both with one command. `train_example.py` is a minimal example of training a policy on the result (see [Example training](#example-training)).

## Quick start

You need [Conda](https://conda-forge.org/download/). Run everything from the repository root.

1. Set up the environment (once). If you have never created it:

   ```bash
   conda env create -f environment.yml        # macOS: environment-macos.yml
   ```

   If you already have `gradus-rl` from before the pipeline existed, update it instead:

   ```bash
   conda env update -f environment.yml --prune
   ```

2. Build the model and check it. This runs both stages and takes about 2 minutes, mostly JAX compiling:

   ```bash
   conda activate gradus-rl
   python -m pipeline.build pipeline/configs/opencr_tdcr_spatial.yaml
   ```

   On a machine without a working NVIDIA GPU, it first prints `Jax plugin configuration error ... CUDA error 303`. That is harmless: JAX falls back to the CPU.

   It worked if every line of the table at the end starts with `PASS`. The robot is in `pipeline/build/opencr_tdcr_spatial/`; open `preview.png` there to see it at rest and with one tendon pulled.

3. Optional: pull the tendons yourself (see [Pulling tendons by hand](#pulling-tendons-by-hand)).

The current model is a stand-in: the open-source [OpenCR LOTR TDCR-spatial](https://github.com/ContinuumRoboticsLab/OpenCR-Hardware/tree/main/mechanics/LOTR_TDCR-spatial) (12 disks, 3 tendons, BSD-3 STLs in `assets/`). Its disks are threaded on a flexible nickel-titanium (NiTi) rod, the backbone. Its config is `configs/opencr_tdcr_spatial.yaml`. In the model, the disks are joined by ball joints whose springiness comes from the backbone's bending stiffness.

## Stage 1: STL to XML

`stl_to_xml.py` writes `build/<name>/robot.xml`, the processed meshes, `features.yaml` (what was detected, for review) and `preview.png`.

- **Automatic:** units (mm vs m), disk orientation and centring, **tendon hole positions** (found by slicing each disk), mass and inertia, joint stiffness from the backbone, and each tendon's length when the chain is straight.
- **From the config:** disk order and spacing, joint pivot/damping/friction/range, backbone tube, servo specs. Guesses are marked `ASSUMED`. If hole detection fails, set `chain.tendon_holes` by hand. Variants can inherit with `extends: other.yaml`.

Each tendon runs from a servo, over a pulley, through the same hole in every disk, and is tied off at the top disk. The servo winds the tendon onto a drum: you command the drum angle in radians, and the servo can pull no harder than its stall torque.

Keep each joint's pivot level with a disk (`pivot_offset: -1`). If pivots sit between disks, every tendon gets shorter as the chain bends, so tight tendons make the chain collapse.

## Stage 2: XML to sim env

`env.py` is the environment. Most people want `TendonLegEnv`: it runs many copies of the robot at once, takes and returns PyTorch tensors, and has the same interface as `go2_env.py`, so it plugs into rsl-rl:

```python
from pipeline.env import TendonLegEnv
env = TendonLegEnv("pipeline/configs/opencr_tdcr_spatial.yaml", num_envs=64)   # builds robot.xml if missing
obs = env.reset()                                  # obs["policy"]: (64, num_obs) tensor
obs, rewards, dones, extras = env.step(actions)    # actions: (64, 3) in [-1, 1]: 0 = just taut, +1 = full wind, -1 = full release
```

Finished episodes restart automatically. Underneath, the simulation runs in JAX; if you write your training in JAX instead, use `JAXTendonLegEnv`, which is one robot written as pure functions for `jax.jit` and `jax.vmap`.

Training works on a CPU but is slow. On a laptop CPU, 64 robots take about 33 environment steps per second while the policy is still pulling tendons at random (up to about 390 when the tendons are relaxed). A million-step PPO run takes about 8 hours. Use a GPU for real runs.

Each step the policy sees joint angles and speeds, servo targets, tendon tensions and the tip position. The example reward is the tip height divided by the chain's straight height, minus a small penalty for pulling hard.

`validate_env.py` checks the environment works: it pulls each tendon and checks the tip bends toward it. It also checks that releasing the servos makes the tendons go slack, that MJX gives the same result as regular CPU MuJoCo, and that 64 copies of the environment run side by side without the numbers blowing up.

## Pulling tendons by hand

Open the model in the MuJoCo viewer (needs a display; works on WSLg):

```bash
python -m mujoco.viewer --mjcf=pipeline/build/opencr_tdcr_spatial/robot.xml
```

1. Press `Shift+Tab` to show the right panel if it is hidden, and open the **Control** section.
2. Drag the `servo0`, `servo1` and `servo2` sliders. Each is a drum angle in radians, from -0.6 (release) to 2.0 (wind); 1 rad pulls 9 mm of cable. Pulling one servo while releasing the other two bends the chain toward the pulled tendon.
3. Use **Clear all** in the Control section to return all servos to 0.

| Input | Action |
|---|---|
| `Space` | Pause / resume |
| `Backspace` | Reset the simulation |
| Left drag / right drag / scroll | Rotate / pan / zoom the camera |
| Double-click a disk, then `Ctrl` + right drag | Push the disk with a force |
| `Tab` / `Shift+Tab` | Toggle the left / right panel |
| `F1` | Show all shortcuts |

## Example training

`train_example.py` is about 50 lines of JAX. It uses evolution strategies, a simple trial-and-error method, to tune a linear policy. It shows how to run many episodes in parallel, not how to train well; for real training use a standard RL algorithm such as PPO on a GPU.

```bash
python -m pipeline.train_example pipeline/configs/opencr_tdcr_spatial.yaml --generations 50
```

On the OpenCR robot the backbone holds the chain upright by itself, so the example task (stand upright) is already solved before training. Expect rewards near 1.0 from the start.

## What the hardware team delivers for the real leg

1. One STL per distinct part, in a shared unit.
2. Link order from base to tip, and the pivot-to-pivot spacing.
3. Pivot position relative to the tendon holes.
4. Servo drum radius, gear ratio, stall torque and max speed. Socket friction if measured.

Then copy a config, fill it in and run `pipeline.build`.

## Known limits

- **Assumed values.** Disk spacing (10 mm, estimated from a photo), the backbone tube size and stiffness, damping, friction and the servo gains are guesses. Each is marked `ASSUMED` in the config.
- **No collisions.** Bending is limited by the joint range instead. Walking will need foot-to-ground contact.
- **MJX vs regular MuJoCo with friction.** With socket friction on, the two drift about 0.2 mm apart in 0.1 s. Measure friction on hardware.
- **CPU speed.** This laptop runs about 1,000 to 1,500 simulation steps per second across 64 environments. Training needs a GPU.
