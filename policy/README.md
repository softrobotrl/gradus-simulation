# SAC for the board-pushing leg

Soft actor-critic (SAC), written from scratch in PyTorch, trained on the board-pushing task from [pipeline/BOARD_TASK.md](../pipeline/BOARD_TASK.md). About 450 lines across four files, written so you can read every line.

Contents: [Quick start](#quick-start) · [Files](#files) · [How SAC works here](#how-sac-works-here) · [The training loop](#the-training-loop) · [Settings](#settings) · [Reading the curves](#reading-the-curves) · [Results so far](#results-so-far) · [Troubleshooting](#troubleshooting) · [Faster: physics on the GPU (WSL2)](#faster-physics-on-the-gpu-wsl2)

## Quick start

On Windows, from the repo root:

```bash
conda env create -f environment-windows.yml    # once; see the note below if conda asks about Terms of Service
conda activate gradus-rl

python -m policy.train_sac --run-name my_run                         # train (Ctrl+C to stop anytime)
tensorboard --logdir logs/sac                                        # second terminal: http://localhost:6006
python -m policy.play_sac logs/sac/my_run/checkpoints/best.pt        # evaluate: prints board speed etc.
python -m policy.play_sac logs/sac/my_run/checkpoints/best.pt --view # watch it in the MuJoCo viewer
python -m policy.train_sac --resume logs/sac/my_run/checkpoints/latest.pt   # continue a stopped run
```

Each run writes to `logs/sac/<run-name>/` (git-ignored): `config.yaml` (every setting used), `metrics.csv` (same numbers as TensorBoard, easy to load in pandas), TensorBoard event files, and `checkpoints/` (`latest.pt` updated every log line, `best.pt` = best average episode return so far, `step_N.pt` every 1 M transitions).

**Conda Terms of Service error.** Recent Miniconda versions refuse `conda env create` until you accept Anaconda's terms for its default channels. Either accept them (`conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`, and the same for `.../pkgs/r` and `.../pkgs/msys2`), or skip the default channels entirely:

```bash
conda create -n gradus-rl --override-channels -c conda-forge python=3.11 numpy pip
conda activate gradus-rl
pip install --extra-index-url https://download.pytorch.org/whl/cu126 "torch>=2.8.0" tensordict tensorboard matplotlib pyyaml "pydantic>=2" mujoco==3.14.0 mujoco-mjx==3.14.0 jax==0.6.2 trimesh networkx scipy
```

Check it: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

## Files

| File | What's in it |
|---|---|
| [replay_buffer.py](replay_buffer.py) | `ReplayBuffer`: a ring of past transitions (obs, action, reward, next obs, terminal) stored as GPU tensors. `add` takes one env step (all envs at once); `sample` draws a random batch. |
| [sac.py](sac.py) | The agent. `ObsNormalizer`, `Actor`, `Critic`, and `SAC` with `act()` and `update()`. `SACHyperparams` holds the network/learning settings. |
| [train_sac.py](train_sac.py) | The training loop, logging and checkpoints. `TrainConfig` holds every setting; each one is also a command-line flag. |
| [play_sac.py](play_sac.py) | Load a checkpoint and evaluate it over many episodes, or watch one env live. |

Two env-side additions made this possible (details in [BOARD_TASK.md](../pipeline/BOARD_TASK.md#what-changed-file-by-file)):

- **[pipeline/cpu_env.py](../pipeline/cpu_env.py)**: a second physics backend. On Windows, JAX can't use the GPU, and MJX on the CPU only manages ~85 transitions/s with random actions. Plain C MuJoCo in 20 threads does ~13,000/s with the same task code. `make_board_env(..., backend="auto")` picks MJX when a GPU is usable and this backend otherwise.
- Running off the end of the board now counts as a **truncation** (like a time-out), not a failure, so SAC keeps valuing the future.

## How SAC works here

SAC learns two things at once:

- **The critic** `Q(s, a)`: "if I'm in state `s`, take action `a`, then follow my policy, how much discounted reward will I get?"
- **The actor** `π(a | s)`: a probability distribution over actions. It's trained to pick actions the critic rates highly, while staying as random as it can afford to (the "soft" part).

The randomness bonus (entropy, weighted by temperature **α**) is what makes SAC explore without hand-tuned noise. α tunes itself: if the policy gets more deterministic than a target, α rises and pushes it back toward randomness, and the other way round.

SAC is **off-policy**: it learns from a replay buffer of past transitions, not only from what the current policy just did. That's why it can reuse each transition many times and why it's sample-efficient.

Mapping to [sac.py](sac.py):

1. **`ObsNormalizer`** ([sac.py:47](sac.py#L47)). The observation mixes millimetres, quaternions and rad/s. The normalizer keeps a running mean and variance of every observation it has seen and rescales inputs to about mean 0, std 1 (clipped to ±5). Raw observations go in the buffer; normalization happens when they're used. The statistics are saved in every checkpoint, since a trained policy only works with them.
2. **`Actor`** ([sac.py:74](sac.py#L74)). An MLP outputs a mean and log-std per servo. It samples `u ~ Normal(mean, std)` and outputs `tanh(u)`, which lands in [-1, 1], the env's action range. The log-probability needs a correction for the tanh squashing ([sac.py:89](sac.py#L89)). Sampling uses the reparameterization trick (`mean + std * noise`), so gradients flow through the sample into the actor. `deterministic=True` returns `tanh(mean)`, used for evaluation and on hardware.
3. **`Critic`** ([sac.py:93](sac.py#L93)). Two independent Q-networks. The training target uses the **smaller** of the two, which counters Q-learning's habit of overestimating.
4. **`SAC.update`** ([sac.py:130](sac.py#L130)), one gradient step on a batch:
   - **Critic** ([line 135](sac.py#L135)): target `y = r + γ (1 − terminal) (min Q_target(s', a') − α log π(a'|s'))`, with `a'` sampled from the current actor at the next state. Both Qs are regressed onto `y` with mean squared error. `terminal` is 1 only for real failures; time-outs and running out of board keep `terminal = 0`, so the value of the next state still counts.
   - **Actor** ([line 145](sac.py#L145)): minimise `α log π(a|s) − min Q(s, a)`, i.e. maximise value plus entropy. The critic is frozen during this step.
   - **α** ([line 153](sac.py#L153)): pushes the policy entropy toward the target −3 (−1 per servo).
   - **Target critics** ([line 159](sac.py#L159)): slow copies of the critics, moved 0.5% toward them each update (`tau`). They make the Bellman target stable instead of chasing itself.

## The training loop

[train_sac.py](train_sac.py), `main()`. Each iteration is one env step for all `num_envs` envs:

1. **Act** ([train_sac.py:184](train_sac.py#L184)). For the first `learning_starts` transitions, uniformly random actions fill the buffer with varied data. After that, actions are sampled from the actor (stochastic, so it keeps exploring).
2. **Store** ([line 195](train_sac.py#L195)). Store `extras["final_observation"]` as the next observation, not the returned `obs`: for envs that just finished, `obs` is already the first observation of a new episode. Store `terminal = done × (1 − time_outs)`. The observation normalizer is updated here too.
3. **Learn** ([line 206](train_sac.py#L206)). `updates_per_step` gradient updates, each on a random batch of `batch_size` transitions.
4. **Log and save** ([line 215](train_sac.py#L215)). Every `log_every` iterations: averages go to TensorBoard and `metrics.csv`, one console line, `latest.pt`, and `best.pt` if the average episode return improved.

At the start, episode clocks are randomised ([line 169](train_sac.py#L169)) so the 256 envs don't all finish and reset on the same step.

The **replay ratio**, how often each transition is used for learning on average, is `updates_per_step × batch_size / num_envs` = 4 × 4096 / 256 = **64** by default.

## Settings

Every field is a command-line flag (`--num-envs 512`, `--rate-cost 20`, ...). `python -m policy.train_sac --help` lists them all. Settings are saved in `config.yaml` and inside every checkpoint; `--resume` reuses them, and flags override.

**Environment and reward** (`TrainConfig`):

| Flag | Default | What it does / when to change it |
|---|---|---|
| `--backend` | auto | `auto`: MJX on a GPU if JAX can use one, else C MuJoCo threads. Force with `mjx` / `cpu`. |
| `--num-envs` | 256 | Parallel envs. More = more data per step; the CPU backend scales to ~512. |
| `--random-direction` | false | `true`: a new push direction each episode (the policy sees it). |
| `--episode-s` | 8.0 | Episode length (s). |
| `--progress-reward` | 100 | Reward per m/s along the direction (1 per cm/s). |
| `--lateral-cost` | 50 | Penalty per m/s sideways. |
| `--rate-cost` | 0.05 | Penalty on servo command changes per step. A large value (20) stops vibration but also stalls SAC's exploration; use `--action-filter` instead. See [Results so far](#results-so-far). |
| `--action-filter` | 0.2 | Servo response per step: 1 = instant, 0.2 ≈ 0.1 s time constant. The default 0.2 stops the 25 Hz vibration exploit (the env itself defaults to 1, i.e. off). |
| `--effort-cost` | 0.01 | Penalty on servo wind-up angle. |
| `--tension-cost` | 0 | Penalty on tendon tension (fraction of the ~100 N max pull, squared). Turn on to protect real cables. |

**Learning** (`TrainConfig` and `SACHyperparams`):

| Flag | Default | What it does / when to change it |
|---|---|---|
| `--total-steps` | 10 M | Transitions to collect. |
| `--learning-starts` | 10,000 | Random-action warm-up. |
| `--buffer-size` | 1 M | Replay capacity (~0.8 GB of GPU memory). Bigger = remembers older behaviour longer. |
| `--batch-size` | 4096 | Transitions per update. On this GPU an update costs the same at 1024 or 4096 (Python and kernel-launch overhead dominate), so big batches are nearly free. |
| `--updates-per-step` | 4 | Gradient updates per env step. More = learns more per transition but runs slower. |
| `--gamma` | 0.99 | Discount. 0.99 looks ~100 steps (2 s) ahead, enough for a ~1 s gait cycle. |
| `--tau` | 0.005 | Target critic tracking speed. |
| `--actor-lr`, `--critic-lr`, `--alpha-lr` | 3e-4 | Learning rates. Lower them if Q values blow up. |
| `--init-alpha` | 0.1 | Starting entropy temperature (it tunes itself). |
| `--hidden`, `--layers` | 256, 2 | Network size. |
| `--grad-clip` | 10 | Max gradient norm per network. |
| `--learner-device` | auto | `cuda` if PyTorch sees a GPU, else `cpu`. |
| `--seed` | 0 | |

## Reading the curves

In TensorBoard (or `metrics.csv`):

| Curve | Meaning | Healthy |
|---|---|---|
| `env/board_speed` | mean board speed along the direction (m/s) | rises from ≈ 0. The hand-written 1 Hz gait does 0.012 |
| `episode/return` | total reward per finished episode | rises |
| `episode/length` | steps per episode (400 = full 8 s) | drops below 400 once it pushes fast enough to reach the end of the board (counts as truncation) |
| `env/lifted` | fraction of time the board is off the rails | > 0: the leg is lifting |
| `env/off_board` | fraction of steps where an episode ran out of board | rises with speed |
| `sac/q` | mean Q estimate | bounded and roughly `reward / (1 − γ)` scale; steady growth to huge values = divergence |
| `sac/alpha` | entropy temperature | falls over training as the policy gets confident |
| `sac/entropy` | policy entropy | moves toward the target (−3) |
| `loss/critic` | Bellman error | noisy, shouldn't explode |
| `time/transitions_per_s`, `time/sim_fraction` | speed, and the share spent in physics vs learning | on this laptop: ~3,600/s, physics ~40%, learning ~60% |

**Always look at the behaviour, not just the reward**: `python -m policy.play_sac <ckpt> --view`. The evaluation also prints physics sanity numbers (max board lift, pad penetration, max tendon tension).

## Results so far

Laptop: RTX 4060 Laptop GPU + 20-core CPU, native Windows, C MuJoCo backend, ~3,600 transitions/s including learning.

**v1, default reward** (`logs/sac/board_v1`, stopped at 2 M transitions, ~8 min). It learned fast: 34 mm/s within 30 k transitions, **123 mm/s** by 1.7 M, every episode reaching the end of the board in 1.4 s. **But it's an exploit**: the policy flips every servo between extremes every 20 ms (25 Hz), hammering the board so it hops forward, even while the tip moves backward. It's real physics *in the sim*, but it relies on servos reversing instantly at full speed and on a smoothness penalty that's tiny next to the progress reward (`rate_cost` 0.05 vs ~11 reward per step). Real servos and cables can't do this; it would not transfer to hardware.

**v2, `--rate-cost 20`** (`logs/sac/board_v2_rate20`, stopped at 320 k). **Didn't learn**: return stuck at −135 and α kept rising. SAC explores by adding fresh random noise to every action, and a large per-step smoothness penalty punishes exactly that noise, so exploring becomes too expensive. Lesson: don't fight vibration with a big `rate_cost`.

**v3, `--action-filter 0.2`** (`logs/sac/board_v3_filter02`, 5 M transitions, 23 min). **A real stepping gait.** The filter makes each servo move only 20% of the way to its new command per step (a ~0.1 s servo response), so 25 Hz vibration can't build up, and exploration noise is smoothed instead of penalised. Learning timeline: beat random at 115 k transitions, matched the hand-written gait (12 mm/s) at 141 k, 100 mm/s at 410 k. Final deterministic evaluation over 64 episodes:

| | |
|---|---|
| board speed | **135 mm/s** (127–144 across episodes) |
| sideways drift | 0.6 mm per episode |
| episodes | all reach the end of the board (17 cm) in ~1.3 s, i.e. truncated |
| gait | ~3.3 Hz cycle: the tip presses into the board and sweeps 5 cm forward carrying it, then drops 1–2 mm below it and swings back along the side |
| physics sanity | max board lift 3.2 mm, pad penetration ≤ 0.6 mm, **tendon tension up to the 100 N servo limit** |

**Still to do before hardware:**
- **Tendon tension.** The policy co-contracts all three tendons near the servo limit to stiffen the leg. Try `--tension-cost` (e.g. 1–5) and check the speed it costs.
- **Real servo numbers.** `action_filter`, the 20 rad/s speed limit and the servo gains are all assumptions. Measure the real servos (step response, max speed) and set them to match.
- **A longer board, or a slower episode.** At 135 mm/s the 40 cm board runs out in 1.3 s, so episodes are short. A longer `BoardConfig.length` lets the policy learn sustained pushing.

v3's settings are now the training defaults: plain `python -m policy.train_sac` reproduces it.

## Troubleshooting

| Problem | Try |
|---|---|
| `sac/q` keeps growing into the hundreds or thousands | first check `env/reward_mean` isn't exploding (an exploit); then lower `--critic-lr` or `--tau` |
| `sac/alpha` collapses toward 0 early and learning stalls | raise `--init-alpha`, or check the policy isn't saturated at ±1 (vibration) |
| Board speed stays ≈ 0 | check `env/lifted`; if the leg never lifts, see the BOARD_TASK.md tuning table. More `--learning-starts` gives more varied early data |
| Fast but weird motion (vibration, bang-bang) | `--action-filter 0.2`, maybe `--tension-cost`; watch with `--view` |
| Slow training | `time/sim_fraction` tells you which side is slow. Learning-bound: fewer `--updates-per-step` or a bigger `--batch-size` with fewer updates. Physics-bound: more `--num-envs`, or WSL2 |
| `Failed to import warp` / `mujoco_warp` printed at start | harmless: MJX checks for NVIDIA Warp, an optional backend that isn't installed |
| `--resume` starts with low return for a while | expected: the replay buffer isn't saved (it's ~1 GB), so it refills from the resumed policy first |

## Faster: physics on the GPU (WSL2)

The code doesn't change: under WSL2 with `environment.yml`, `--backend auto` picks MJX on the GPU and the learner stays on the GPU. See [BOARD_TASK.md, "Running on your GPU"](../pipeline/BOARD_TASK.md#running-on-your-gpu) for setup. On this laptop, learning already takes ~60% of each iteration, so the bigger win is speeding up the update (e.g. CUDA graphs, since each update is dominated by kernel-launch overhead) rather than the physics.
