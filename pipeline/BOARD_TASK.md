# Board-pushing task (single leg)

A training environment where the upward-facing tendon leg pushes a wooden board forward. It is the RL team's proof-of-concept task while the real soft-robot leg is being built: the same "use friction at the foot to move something" skill as walking, but with the leg upside down and the board playing the ground.

- Code: [board_env.py](board_env.py) (the task), [env.py](env.py) (shared machinery, lightly refactored), [cpu_env.py](cpu_env.py) (fast CPU physics backend), [validate_board_env.py](validate_board_env.py) (proves the task is solvable).
- For SAC: the trainer is in [policy/](../policy/README.md). In your own code, use `make_board_env(...)` and read [Using it from SAC](#using-it-from-sac) for the two extras SAC needs.

Contents: [The task](#the-task) · [Quick start](#quick-start) · [What changed, file by file](#what-changed-file-by-file) · [How the sim is built](#how-the-sim-is-built) · [Observations](#observations) · [Reward](#reward) · [Every setting](#every-setting) · [Using it from SAC](#using-it-from-sac) · [Running on your GPU](#running-on-your-gpu) · [How to extend](#how-to-extend) · [Troubleshooting](#troubleshooting) · [Known limits](#known-limits)

## The task

```
 side view (x = push direction →)

        ┌──────────────────────── board (40 x 15 x 1.2 cm, 0.1 kg) ───────────────────────┐
  rail ═╧═══════════════════════════════════════════════════════════════════════════════╧═ rail
                                           ● ← pad (rubber sphere on the top disk)
                                           ┃
                                           ┃  leg: 12 disks on a NiTi backbone,
                                           ┃  3 tendons pulled by 3 servos
                                           ┃
                                     ══════╩══════ base
```

The board rests on rails. The straightened leg reaches **3 mm above** the rails (`rail_gap`), so when the leg is straight it lifts the board slightly off them; when the leg bends, the tip drops and the board sits on the rails again.

**Why rails?** The leg can only move the board through friction at its tip. If the board rested only on the leg, tilting the tip forward would carry the board forward and tilting back would carry it straight back: no net travel. With rails, the leg can do a stepping gait, exactly like a walking foot:

1. **Stance**: the leg is near straight, the pad is pressed up into the board (board lifted off the rails), and the tip sweeps forward, dragging the board with it.
2. **Swing**: the leg bends, the tip drops below the board, the board settles on the rails (rail friction holds it), and the tip swings back underneath.

`validate_board_env.py` checks this works with a hand-written gait before anyone spends GPU time: on the current settings the scripted gait pushes the board about **12 mm/s**, either direction, and random actions push it nowhere on average.

## Quick start

Run everything from the repository root, in the `gradus-rl` conda env.

```bash
# 1. Prove the task is solvable (and see how fast your machine runs it). ~4 min on a laptop CPU, mostly compiling.
python -m pipeline.validate_board_env pipeline/configs/opencr_tdcr_spatial.yaml

# 2. Look at it: the validate step writes the whole scene to board_scene.xml.
python -m mujoco.viewer --mjcf=pipeline/build/opencr_tdcr_spatial/board_scene.xml
```

In the viewer, open the **Control** panel (`Shift+Tab`) and drag `servo0..2`. Pulling one servo while releasing the other two bends the leg toward that tendon and drops the board onto the rails; set all to 0 and the leg straightens and lifts it.

```python
# 3. Use it from Python
import torch
from pipeline.board_env import make_board_env

env = make_board_env("pipeline/configs/opencr_tdcr_spatial.yaml", num_envs=64)  # MJX on GPU, else C MuJoCo
obs = env.reset()["policy"]                     # (64, 99) float32 tensor on env.device
for _ in range(50):
    actions = torch.rand(64, env.num_actions, device=env.device) * 2 - 1   # 3 servos, each in [-1, 1]
    obs, reward, done, extras = env.step(actions)
    obs = obs["policy"]
print(extras["log"])     # {'/reward_mean': ..., '/board_speed': ..., '/lateral_speed': ..., '/lifted': ..., '/off_board': ...}
print(env.obs_layout)    # which observation columns mean what
```

## What changed, file by file

### `pipeline/env.py` (edited; the upright example task behaves as before)

The old `JAXTendonLegEnv` hard-coded one task (keep the chain upright) inside `__init__`, `reset` and `step`. To let a new task reuse everything without copying it, those were split into small **hooks** that a subclass overrides. Nothing about the physics or the upright task's numbers changed; `validate_env.py` still passes.

| Hook | What it does | Default (upright task) | Board task overrides it to... |
|---|---|---|---|
| `_load_model(xml_path)` | Compile the scene | load `robot.xml` as-is | add a pad and a board via `MjSpec` |
| `_reset_scene(key, qpos)` | Starting pose of added objects | nothing | place the board at a random offset |
| `_init_info(key, data)` | Per-env task data at reset | `{}` + bookkeeping | push direction, board position |
| `_task(state, data, target)` | (reward, failed, truncated, info) after a step | tip height − effort, never ends early | board progress reward; truncated when the board runs out |
| `_obs(data, target, info)` | Observation vector | qpos, qvel, targets, tensions, tip | robot-only qpos/qvel + board terms |

Before / after of `step`:

```python
# before: everything inline
def step(self, state, action):
    ...servo rate limit, physics substeps...
    upright = (data.site_xpos[self.tip, 2] - self.base_z) / self.height
    reward = upright - self.effort_cost * effort
    ...done on time limit or NaN...

# after: shared part + task hook
def step(self, state, action):
    target = self.servo_targets(state, action)           # action -> rate-limited servo setpoints
    data = ...physics substeps (unchanged code)...
    return self.finish_step(state, data, target)         # reward, done, obs: shared by both backends

def finish_step(self, state, data, target):
    reward, failed, truncated, info = self._task(state, data, target)  # task-specific
    ...done on time limit, `truncated`, NaN, or `failed`...
```

`step` is split into `servo_targets` (action to rate-limited setpoints), the physics, and `finish_step` (reward, done, obs) so a second physics backend can reuse the exact same task code. `reset` is split the same way, into `reset_qpos` and `finish_reset`. See `cpu_env.py` below.

**`failed` vs `truncated`.** Both end the episode. `failed` means "the future is worth nothing" (for now, only NaN physics), so SAC doesn't bootstrap. `truncated` means "ended for a reason outside the task", like the time limit, so SAC still bootstraps. Running out of board is a truncation: pushing far is the goal, and the policy can't see the board edges. It was a failure in the first version; training showed that confuses the critic once the policy gets fast.

Other changes in `env.py`:

- **`State.info`** (new field): a dict carried from step to step. It always has `"terminated"` (1 if the episode ended by failure rather than the time limit or a truncation) and `"metrics"` (per-env numbers to log). The board task also keeps the push direction and the board's previous position there (to compute its speed).
- **`TendonLegEnv` returns two new extras**, both needed for correct SAC targets (see [Using it from SAC](#using-it-from-sac)):
  - `extras["final_observation"]`: each env's observation right after this step, *before* auto-reset replaced finished envs with fresh ones.
  - `extras["time_outs"]` now excludes failures: it is 1 when the episode ended by the time limit or a truncation (e.g. ran out of board).
- **`extras["log"]`** now also averages every entry of `info["metrics"]` (e.g. `/board_speed`), so rsl-rl/TensorBoard pick them up automatically.
- **Device choice** (`device="auto"|"gpu"|"cpu"`, new `pick_device` function): see [Running on your GPU](#running-on-your-gpu).
- **`TendonLegEnv.jax_env_cls`** (class attribute): which JAX env the batched wrapper builds. `BoardPushEnv` is just `TendonLegEnv` with this set to `JAXBoardPushEnv`.
- `_to_jax` moves incoming torch tensors to the env's device first, so passing CPU actions to a GPU env just works.
- `obs_size` is now found with `jax.eval_shape` (traces shapes only, runs no physics).
- **`action_filter`** (new setting, default 1 = off): each step the servo command moves only this fraction of the way to the new action, a first-order servo response. 0.2 ≈ a 0.1 s time constant (~2 Hz). Added after the first SAC run found a 25 Hz vibration exploit; see [policy/README.md](../policy/README.md#results-so-far).
- `gpu_problem()` (new): returns why the GPU can't be used, or None. Shared by `pick_device` and `make_board_env`.

### `pipeline/cpu_env.py` (new)

A second physics backend, `CpuTendonLegEnv`, with the same PyTorch interface. MJX is built for GPUs, so on a CPU it's slow: ~85 transitions/s with random actions on this laptop. Plain C MuJoCo, stepping each env in a thread pool (MuJoCo releases Python's GIL while it simulates), does ~13,000 transitions/s on 20 cores.

Only the physics is different. Every step:

1. The JAX `servo_targets` turns actions into setpoints.
2. The threads run `mj_step` for 20 substeps per env and copy the fields task code reads into a `SimData` (qpos, qvel, actuator_force, site_xpos, geom_xpos).
3. The JAX `finish_step` computes reward, done and obs.

Finished envs are reset with `reset_qpos`, then `mj_forward`, then `finish_reset`. So rewards and observations come from the same code as the MJX env. The validator checks that the scripted gait moves the board the same distance on both backends (50.7 vs 50.8 mm). If MuJoCo detects a diverged simulation, it resets that env by itself and records a warning; the backend turns that warning into a NaN, so the task code marks the episode as failed.

**Rule for task code:** to run on both backends, `_task`, `_obs` and `_init_info` may only read the `SimData` fields from `data`.

### `pipeline/board_env.py` (new)

- `BoardConfig`: every physical number of the board, rails and pad ([table](#every-setting)).
- `JAXBoardPushEnv(JAXTendonLegEnv)`: the task, as the hooks above. Also `obs_layout` (name → column slice) and `write_scene_xml()` (dump the scene for the viewer).
- `BoardPushEnv(TendonLegEnv)`: the batched PyTorch version on MJX (fast on a GPU).
- `BoardPushEnvCPU(CpuTendonLegEnv)`: the same, with C MuJoCo physics in CPU threads (fast without a GPU).
- `make_board_env(config, num_envs, backend="auto")`: picks MJX if JAX and PyTorch can both use a GPU, otherwise the CPU backend. Prints which one it chose.

### `pipeline/validate_board_env.py` (new)

The acceptance test for the task, and for the CPU backend too. Its checks are listed in the file docstring; the key one is the scripted gait, which shows the default settings actually let a leg push the board. If you change `BoardConfig` (heavier board, different rail gap...), rerun it: if the scripted gait stops moving the board, the RL agent won't learn to either.

### Docs

This file, plus short pointers in [pipeline/README.md](README.md) and the top-level [README.md](../README.md). `stl_to_xml.py`, `robot.xml` and the robot config are untouched: the board is added on top of the generated robot at load time, so the hardware pipeline stays task-agnostic.

## How the sim is built

`JAXBoardPushEnv._load_model` loads `robot.xml` into a `mujoco.MjSpec` (an editable version of the model), adds three things, and compiles:

1. **Pad**: a 4 mm sphere geom attached to the top disk at the `tip` site. The real top disk is flat; a sphere is used because sphere-against-box contact is cheap and robust in MJX (mesh contact is slow there).
2. **Board**: a box geom on a new body. It has three **slide joints** (`board_x`, `board_y`, `board_z`) and no rotation: it can translate but never tilt, which stands in for guides on the rig.
3. **Rails** are not geoms. `board_z`'s lower joint limit is 0, and z = 0 means "resting on the rails". That is much cheaper than simulating rail contacts and behaves the same for a board that cannot tilt.

Plain-words glossary for the settings involved:

- **`contype` / `conaffinity`**: MuJoCo collides two geoms only if one's `contype` shares a bit with the other's `conaffinity`. Every robot geom has both at 0 (no collisions, as before). The pad has `contype=1`, the board `conaffinity=1`, so the pad-board pair is the only contact in the whole scene.
- **`friction`** (0.8): the sliding friction coefficient at the pad-board contact. This is what lets the leg drag the board.
- **`solref` time constant** (`contact_timeconst`, 5 ms): how stiff the contact is. MuJoCo's default (20 ms) let the pad sink 1.6 mm into the board under its weight, which is half the rail gap; 5 ms gives about 0.2 mm.
- **`frictionloss`** on `board_x`/`board_y` (`rail_friction`, 0.2 N): a constant resisting force when the board slides, i.e. rail friction. It holds the board in place while the tip swings back. If it's bigger than what the pad's friction can deliver (about μ × board weight = 0.8 N), the leg can't push at all.
- **`damping`** (`rail_damping`): stops the board coasting after a push.

## Observations

99 numbers per env, in this order. `env.obs_layout` gives the slices in code (e.g. `obs[:, env.obs_layout["board_vel"]]`). Units are SI; the values span very different scales (mm vs rad/s), so **normalize observations** in the agent (a running mean/std, as rsl-rl's `empirical_normalization` does).

| Columns | Name | Meaning |
|---|---|---|
| 0:47 | `robot_qpos` | 11 ball-joint quaternions (4 each) + 3 cable slide positions (m) |
| 47:83 | `robot_qvel` | 11 ball-joint angular velocities (3 each, rad/s) + 3 cable speeds (m/s) |
| 83:86 | `servo_target` | servo angle setpoints after rate limiting (rad) |
| 86:89 | `tendon_tension` | tendon tension (N), what motor current would measure |
| 89:92 | `tip_pos` | tip position (m); the leg base is at the origin |
| 92:95 | `board_vel` | board velocity x, y, z (m/s) |
| 95:96 | `board_lift` | board height above the rails (m); 0 = resting on them |
| 96:97 | `pad_gap` | distance from the pad top to the board underside (m); ≤ 0 means touching |
| 97:99 | `direction` | unit vector of the push direction; (1, 0) unless `random_direction=True` |

The board's absolute x/y position is deliberately left out: it grows without bound as the board moves and tells the policy nothing it needs.

## Reward

Per control step (every 20 ms):

```
reward =  progress_reward * v_along           # board speed along the push direction (m/s)
        - lateral_cost    * |v_across|        # board speed sideways (m/s)
        - effort_cost     * mean((servo_target / max_angle)^2)
        - rate_cost       * mean((servo_change / max_change_per_step)^2)
```

| Term | Default | Typical size | Why |
|---|---|---|---|
| `progress_reward` | 100 (per m/s, so 1 per cm/s) | ~1.2 per step for the scripted gait | the goal |
| `lateral_cost` | 50 | 0 if pushing straight | don't push the board sideways |
| `effort_cost` | 0.01 | ≤ 0.01 | don't hold tendons wound for no reason (inherited from the upright task) |
| `rate_cost` | 0.05 | ≤ 0.05 | smooth servo commands, which transfer better to hardware. Too small to stop vibration exploits; see `action_filter` |
| `tension_cost` | 0 (off) | up to the weight | `mean((tension / max pull)²)`; turn on to keep tendon loads realistic (an unpenalized policy uses the full ~100 N) |

**Episode ends** in one of three ways:
- **Time-out**: after 8 s (400 steps).
- **Truncation** (bootstrapped like a time-out): the board has moved so far that the leg is within `edge_margin` (3 cm) of a board edge, i.e. 17 cm forward or 4.5 cm sideways.
- **Failure**: only if the physics blows up (NaN).

A trained policy pushing at several cm/s reaches the front edge within a few seconds, so most of its episodes end by truncation (the `/off_board` metric).

**If training does X, change Y:**

| Symptom | Try |
|---|---|
| Board slides sideways / curves | raise `lateral_cost` |
| Servos chatter, jerky motion, 25 Hz "vibration" pushing | set `action_filter` (e.g. 0.2). A large `rate_cost` also stops it, but fights SAC's exploration noise and can stall learning |
| Leg never lifts the board (`/lifted` ≈ 0) | lower `rail_gap` or the board `mass` |
| Board pushed, then slides back | raise `rail_friction` or `rail_damping` |
| Leg can't move the board at all | lower `rail_friction` below ~μ × board weight; rerun the validator |
| Policy learns nothing early | progress reward is tiny at first; consider a shaped bonus for `pad_gap` ≤ 0 (contact) |

## Every setting

**`BoardConfig`** (pass as `board=BoardConfig(mass=0.2, ...)`). ASSUMED = a guess until the real rig is measured.

| Field | Default | Units | ASSUMED? | Effect |
|---|---|---|---|---|
| `length` | 0.40 | m | rig | board size along x; also how far it can travel before the episode is truncated |
| `width` | 0.15 | m | rig | along y |
| `thickness` | 0.012 | m | rig | visual mostly |
| `mass` | 0.1 | kg | yes | heavier = more pad friction available but more load on the leg |
| `rail_gap` | 0.003 | m | rig | how high the straight leg reaches above the rails; bigger = easier to lift, but the leg must bend more to let go |
| `rail_friction` | 0.2 | N | yes | resists the board sliding on the rails |
| `rail_damping` | 0.5 | N·s/m | yes | stops coasting |
| `max_lift` | 0.03 | m | - | upper limit of the board's z travel |
| `pad_radius` | 0.004 | m | yes | size of the contact sphere on the tip |
| `friction` | 0.8 | - | yes | pad-on-board friction coefficient |
| `contact_timeconst` | 0.005 | s | - | contact stiffness; keep ≥ 2 × sim timestep (2 ms) |
| `start_offset` | 0.02 | m | - | random board start offset in x and y |
| `edge_margin` | 0.03 | m | - | episode ends (truncated) when the leg gets this close to a board edge |

**`JAXBoardPushEnv` / `BoardPushEnv` keyword arguments:**

| Argument | Default | Effect |
|---|---|---|
| `num_envs` | (required, batched envs only) | parallel envs |
| `seed` | 0 | RNG seed (batched envs only) |
| `backend` | `"auto"` | `make_board_env` only: `"auto"`, `"mjx"`, `"cpu"` |
| `device` | `"auto"` | `BoardPushEnv` (MJX) only: `"auto"`, `"gpu"`, `"cpu"`; see [Running on your GPU](#running-on-your-gpu) |
| `num_threads` | CPU cores | `BoardPushEnvCPU` only: physics threads |
| `board` | `BoardConfig()` | physical setup |
| `random_direction` | False | True: new random push direction each episode (shown in the obs) |
| `progress_reward` | 100.0 | reward per m/s along the direction |
| `lateral_cost` | 50.0 | penalty per m/s sideways |
| `rate_cost` | 0.05 | servo smoothness penalty |
| `tension_cost` | 0.0 | tendon tension penalty |
| `action_filter` | 1.0 | servo response per step (1 = instant; 0.2 ≈ 0.1 s time constant) |
| `effort_cost` | 0.01 | servo effort penalty |
| `episode_s` | 8.0 | episode length in seconds (`max_episode_length` = episode_s / 0.02) |
| `init_bend_deg` | 3.0 | random starting bend per joint (degrees) |
| `init_servo_angle` | -0.3 | starting servo angle (rad; negative = tendons slack) |
| `rebuild` | False | regenerate `robot.xml` from the STLs first |

Actions: 3 numbers in [-1, 1], one per servo. 0 = drum angle 0 (tendon just taut), +1 = fully wound (2.0 rad), -1 = fully released (-0.6 rad). The env rate-limits each servo to 20 rad/s, so a step can move a setpoint by at most 0.4 rad. Tanh-squashed SAC actions fit directly.

## Using it from SAC

Two details matter for any off-policy agent:

1. **The next observation.** Finished envs auto-reset inside `step`, so for those envs the returned `obs` is already the *first* observation of a new episode. Storing that as `next_obs` would teach the critic garbage. `extras["final_observation"]` holds every env's real post-step observation (for envs that didn't finish, it's identical to `obs`), so just always store it.
2. **When to bootstrap.** A time-out (or running out of board) is not a real end: the board could have kept moving, so the critic should still bootstrap from the next state. Only failures (`done` and not `time_outs`) are terminal.

[policy/train_sac.py](../policy/train_sac.py) does exactly this; the snippet below is its core.

```python
import torch
from pipeline.board_env import make_board_env

env = make_board_env("pipeline/configs/opencr_tdcr_spatial.yaml", num_envs=256)
dev = env.device                                    # "cuda:0" or "cpu": put networks and buffer here too
obs = env.reset()["policy"]

for it in range(num_iterations):
    with torch.no_grad():
        action = actor.sample(obs)                  # (num_envs, 3) in [-1, 1]
    next_obs, reward, done, extras = env.step(action)
    next_obs = next_obs["policy"]

    real_next = extras["final_observation"]         # post-step obs, even for envs that just reset
    terminal = done.float() * (1 - extras["time_outs"])  # 1 only for failures
    buffer.add(obs, action, reward, real_next, terminal)  # target: r + gamma * (1 - terminal) * V(real_next)

    obs = next_obs                                   # keep acting from the fresh obs
    # ... SAC update from buffer, log extras["log"] ...
```

Keep the replay buffer as torch tensors on `env.device`; the env's tensors already live there, so nothing passes through the CPU.

## Running on your GPU

The env runs on a GPU if it can and falls back to the CPU if it can't. When it starts it prints which one it chose:

```
BoardPushEnv: running on GPU (NVIDIA GeForce RTX 4060)
BoardPushEnv: running on CPU (JAX found no usable GPU)        # plus a warning
```

`device="auto"` (default) picks the GPU only if both JAX and PyTorch can use it (otherwise data would be split across devices). `device="gpu"` raises an error instead of falling back; `device="cpu"` always uses the CPU. `validate_board_env.py` takes the same `--device` flag.

**Windows catch:** JAX (which runs the MJX physics) has no CUDA build for native Windows, so on plain Windows `auto` always picks the CPU. To train on an NVIDIA card from a Windows PC, use **WSL2**:

1. In PowerShell (admin): `wsl --install -d Ubuntu`, then reboot. The normal Windows NVIDIA driver is all you need; don't install a Linux driver inside WSL.
2. In Ubuntu: install [Miniforge](https://conda-forge.org/download/), clone the repo, then `conda env create -f environment.yml` and `conda activate gradus-rl`.
3. Check: `nvidia-smi` shows the card, and `python -c "import jax; print(jax.devices())"` prints `[CudaDevice(id=0)]`.
4. Run `python -m pipeline.validate_board_env pipeline/configs/opencr_tdcr_spatial.yaml --device gpu --sizes 256 1024 2048`. The last lines show steps per second and memory per batch size; pick the largest `num_envs` that fits comfortably in memory (8 GB on an RTX 4060, shared with the SAC networks and buffer).

JAX is told not to grab most of the GPU memory at start-up (`XLA_PYTHON_CLIENT_PREALLOCATE=false` in `env.py`), so JAX and PyTorch share the card.

Measured so far:

| Machine | num_envs | Env steps/s | Notes |
|---|---|---|---|
| Laptop CPU (20 cores), native Windows, **MJX** | 64 / 512 | ~21 / ~85 with random actions | ~660 with relaxed servos; MJX is built for GPUs |
| Laptop CPU (20 cores), native Windows, **C MuJoCo backend** | 256 / 512 | ~11,000 / ~13,000 with random actions | `make_board_env` picks this when there's no usable GPU |
| RTX 4060 (WSL2) | 256 / 1024 / 2048 | *run step 4 and fill in* | |

## How to extend

**Arbitrary push direction.** Already built in: `BoardPushEnv(..., random_direction=True)`. Each episode draws a random direction; the policy sees it as the last two observation columns and is rewarded for pushing along it. The validator already checks a gait rotated 90° pushes sideways. For sideways directions the board's width (15 cm) limits how far it can travel before the episode is truncated, so consider a square board: `board=BoardConfig(length=0.4, width=0.4)`.

**The real leg.** When hardware delivers the Gradus leg, write its robot config (see [README.md](README.md#what-the-hardware-team-delivers-for-the-real-leg)), build it, and pass that config path to `BoardPushEnv`. The board is added to whatever robot the config builds, as long as the robot has a `tip` site on its last body. Re-measure and update `BoardConfig` on the real rig, then rerun the validator.

**A new task.** Copy the pattern in `board_env.py`: subclass `JAXTendonLegEnv`, set your own attributes *before* calling `super().__init__` (it compiles the model and traces `reset`), override the hooks you need, and make two-line `TendonLegEnv` / `CpuTendonLegEnv` subclasses that set `jax_env_cls`. To work on the CPU backend, read only the `SimData` fields from `data`. `_init_info` and `_task` must return dicts with the same keys and array shapes, because finished envs are swapped for fresh ones element by element.

## Troubleshooting

| Problem | Cause / fix |
|---|---|
| `Jax plugin configuration error ... CUDA error 303` | JAX can't reach a GPU and falls back to the CPU. Harmless; see [Running on your GPU](#running-on-your-gpu). |
| Very slow on CPU, long pause at start | You're on the MJX backend without a GPU. Use `make_board_env` (or `BoardPushEnvCPU`), which is ~150× faster on a CPU. The first MJX step also compiles for ~1 min. |
| `done` with `time_outs` = 0 | A failure: the physics went NaN (or MuJoCo reported divergence). Those envs return reward 0 and are reset. |
| Board jumps or flies up | Contact too stiff for the timestep: keep `contact_timeconst` ≥ 0.002 s. |
| Scripted gait check fails after changing `BoardConfig` | The new settings make pushing impossible or very hard (too heavy, too much rail friction, too little rail gap). Fix that before training. |
| `TypeError` about dict keys in `jax.tree.map` | A custom task's `_init_info` and `_task` return different keys or shapes. |

## Known limits

- **Assumed values.** Board mass, friction coefficients and rail friction are guesses; measure them on the real rig.
- **No tilting.** The board can't rotate. A real board pushed off-centre may yaw; add a hinge joint about z if the rig shows it.
- **Constant rail friction.** `frictionloss` resists sliding with the same force whether the board is on the rails or lifted off them. Real rail friction disappears when the board is lifted, so the sim is slightly harder than reality.
- **Sim-to-real observations.** The observation includes ball-joint angles and board height/velocity, which hardware may not measure directly. A "real sensors only" observation mode is a later step.
- **Sphere pad.** The real tip is a flat disk; the sphere touches at one point.
