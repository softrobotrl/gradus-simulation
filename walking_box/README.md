# Walking box

A simple box-bodied quadruped built from four of the tendon legs. Flat floor,
forward walking only, PPO, no surrogate model.

```
        trunk box
   FL ─────────── FR
    │             │
   BL ─────────── BR
```

- 4 legs × 8 tendon servos = 32 actuators.
- **Diagonal trot nominal**: the FL+BR diagonal swings while FR+BL is in
  stance, then the pairs swap. During swing, the back tendons pull the foot
  rearward and lift it. A stance-phase back-tendon pull can also push the body
  forward; `swing_frac < 0.5` adds double support. Pulling front and back
  together cancels propulsion because both directions shorten the leg.
- **32 policy actions**: one residual tendon pull per actuator. The policy can
  correct every leg independently around the nominal trot.
- 120 observations: trunk quat, body velocity, angular velocity, height error,
  command, gait phase, 32 tendon lengths, 32 tendon forces, 4 foot heights,
  4 foot vertical velocities, previous 32 actions.
- CPU MuJoCo + `rsl-rl` PPO; no MJX/Warp required.
- Servos are stronger than the original single-leg model: `kp=8000`,
  `kv=80`, still limited to 2 N·m / 200 N and 31.4 mm stroke.

## Build

```bash
set -a && . ./.env && set +a
python -m mujoco_to_mjcf.build --out urdf/gradus/walk   # tip-down leg + tip site
python -m walking_box.build_quadruped                   # 4 legs + trunk + feet
python -m walking_box.build_phys                        # fast physics-only XML
```

- `urdf/gradus/walk/quadruped.xml` — full visual/collision model for rendering.
- `urdf/gradus/walk/quadruped_phys.xml` — 6 collision geoms (floor, trunk,
  4 feet), explicit inertials, same mass distribution. Used for training.
- Both are gitignored.

## Train

```bash
python -m walking_box.train_rsl \
    --num-envs 64 --iterations 1500 --steps-per-env 24 \
    --num-threads 16 --episode-steps 400
```

Curriculum:

1. 50 iterations with the nominal gait off, `cmd_max=0` (stand)
2. 300 at `cmd=0.05..0.15` m/s with the diagonal trot on
3. 1150 at `cmd=0.10..0.25` m/s

Commands are kept strictly positive during walking to discourage a standing
local optimum.  Stability-heavy fine-tunes can still collapse to standing if
the velocity/progress weights are too low; the current robust fine-tune keeps
`vel_weight=3.0` and `progress_weight=1.0`.

The policy is a 256×256 MLP actor/critic; PPO with 4 epochs, 4 minibatches,
`lr=3e-4`, adaptive KL. Checkpoints go to
`logs/walking_box/<run>/model_*.pt`. Torch and NumPy are seeded from
`EnvConfig.seed`, so repeated runs are reproducible.

To fine-tune a checkpoint without replaying the stand/walk curriculum:

```bash
python -m walking_box.train_rsl \
    --resume logs/walking_box/walk_diag_strong/model_1497.pt \
    --iterations 800 --num-envs 64 --steps-per-env 24 \
    --num-threads 8 --episode-steps 400 --run-name walk_diag_yaw
```

The current robust gait starts from the earlier robust checkpoint with the
actor output layer zeroed, so training begins exactly on the new nominal:

```bash
python -m walking_box.train_rsl \
    --resume logs/walking_box/walk_diag_robust/model_2100.pt \
    --num-envs 64 --iterations 500 --steps-per-env 24 \
    --num-threads 8 --episode-steps 400 --run-name walk_diag_robust_fast \
    --entropy-coef 0 --learning-rate 2e-4 --std 0.15 --zero-actor-output \
    --fall-penalty 5 --tilt-weight 3 --reset-noise 0.0 \
    --gait-amp 0.7 --period 0.50 --swing-frac 0.5 --stance-amp 0.1 \
    --residual-scale 0.3 --cmd-min 0.10 --cmd-max 0.15 \
    --vel-weight 3.0 --vel-sigma 0.15 --progress-weight 1.0 \
    --slip-weight 0.2 --lat-weight 1.0 --yaw-weight 0.8
```

`--zero-actor-output` zeroes the last actor layer, so the first rollout is the
nominal gait rather than the old policy's residuals.  `--swing-frac`,
`--stance-amp`, and `--residual-scale` control the nominal trot and how much
authority the policy has around it.  The checkpoint is evaluated with
`--residual-scale 0.35` at `cmd=0.18`.

For robust fine-tuning, add `--reset-noise 1.0` to perturb each reset by
±20 mm in xy, ±0.12 rad yaw, ±0.03 rad roll/pitch, and ±0.02 m/s base
velocity.  `--gait-amp`, `--period`, `--swing-frac`, `--stance-amp`, and
`--residual-scale` override the nominal gait and residual authority used
during the fine-tune.

## Evaluate / render

```bash
python -m walking_box.evaluate \
    --ckpt logs/walking_box/walk_diag_robust_fast/model_2550.pt \
    --cmd 0.18 --gait-amp 0.7 --period 0.50 \
    --swing-frac 0.5 --stance-amp 0.1 --residual-scale 0.35 --steps 400
```

Writes `eval/metrics.txt` and `eval/walk.gif` (3/4 view, 25 fps at
`--every 4`). The camera follows the trunk so the gait stays framed. Add
`--no-render` to write metrics only.

To check sensitivity to the initial pose, run the same policy from perturbed
resets:

```bash
python -m walking_box.robustness \
    --ckpt logs/walking_box/walk_diag_robust_fast/model_2550.pt \
    --cmd 0.18 --gait-amp 0.7 --period 0.50 \
    --swing-frac 0.5 --stance-amp 0.1 --residual-scale 0.35 --seeds 20
```

The default perturbation is ±20 mm in xy, ±0.12 rad yaw, ±0.03 rad
roll/pitch, and ±0.02 m/s base velocity.

### Result (walk_diag_robust_fast, 500 iterations)

The earlier robust policy was safe but slow (0.069 m/s).  Two problems kept it
there: the nominal trot had no stance-phase pull, and the stance-slip penalty
was effectively unbounded.  On the faster nominal the raw slip term reached
−132, dwarfing every other reward term.  This run changes both:

- the nominal trot uses `amp=0.7`, `period=0.50`, `swing_frac=0.5`,
  `stance_amp=0.1`; the stance-phase pull pushes the body forward instead of
  leaving the stance diagonal passive;
- the slip penalty is contact-aware and capped, so it can no longer dominate
  the reward;
- the actor's last layer is zeroed, so fine-tuning starts on the stable
  nominal instead of fighting the old policy.

Best checkpoint: `walk_diag_robust_fast/model_2550.pt`.  Evaluated at
`cmd=0.18`, `gait_amp=0.7`, `period=0.50`, `swing_frac=0.5`,
`stance_amp=0.1`, `residual_scale=0.35`, 400 steps (4 s):

| metric | result |
|---|---|
| mean forward speed | 0.131 m/s |
| distance | 0.526 m |
| lateral drift | −0.095 m |
| yaw | −0.10 rad |
| trunk height | 0.656 m |
| minimum trunk height | 0.644 m |
| minimum up-vector | 0.996 |
| fall | no |

Across 20 perturbed initial states, all 20 survive the full 4 s; mean forward
distance is 0.144 m.  The perturbed starts are deliberately harsher than the
nominal demo; the policy stays upright, but some starts settle into a slower
limit cycle.

Command sweep on the same checkpoint:

| command | mean vx | distance | lateral drift | yaw | fall |
|---|---|---|---|---|---|
| 0.12 | 0.114 | 0.456 m | −0.124 m | −0.11 | no |
| 0.15 | 0.121 | 0.485 m | −0.091 m | −0.24 | no |
| **0.18** | **0.133** | **0.526 m** | **−0.095 m** | **−0.10** | **no** |
| 0.20 | 0.129 | 0.514 m | −0.121 m | −0.17 | no |

A more aggressive nominal (`amp=1.1`, `period=0.50`, `stance_amp=0.2`)
walks 0.108 m/s open-loop and trained to 0.123 m/s, but only 4/20 perturbed
starts survived.  It was rejected in favor of the robust nominal above.  A
300-iteration straightness fine-tune with stronger heading/lateral weights
over-corrected into a standing local optimum, so `model_2550` is kept.

### Earlier robust result (walk_diag_robust, 800 iterations)

The robust fine-tune resumed from `walk_diag_yaw/model_1600.pt` with
reset-state randomization and a cleaner/slower nominal gait:

```bash
python -m walking_box.train_rsl \
    --resume logs/walking_box/walk_diag_yaw/model_1600.pt \
    --num-envs 64 --iterations 800 --steps-per-env 24 \
    --num-threads 8 --episode-steps 400 --run-name walk_diag_robust \
    --entropy-coef 0 --learning-rate 1e-4 --std 0.2 \
    --fall-penalty 5 --tilt-weight 3 --reset-noise 1.0 \
    --gait-amp 0.8 --period 0.86
```

Best robust checkpoint: `model_2100.pt`, evaluated at `cmd=0.18 m/s`,
`gait_amp=0.8`, `period=0.86`, 400 steps (4 s):

| metric | result |
|---|---|
| mean forward speed | 0.069 m/s |
| distance | 0.275 m |
| lateral drift | −0.006 m |
| yaw | −0.01 rad |
| trunk height | 0.639 m |
| minimum trunk height | 0.618 m |
| minimum up-vector | 0.995 |
| fall | no |

Across 20 perturbed initial states, all 20 survive the full 4 s; mean forward
distance is 0.299 m. The walk is slow but stays upright and nearly straight.

Speed/robustness trade-off on the same checkpoint and gait:

| command | nominal speed | distance | 20-seed survival |
|---|---|---|---|
| **0.18** | **0.069 m/s** | **0.275 m** | **20/20** |
| 0.20 | 0.078 m/s | 0.311 m | 19/20 |
| 0.25 | 0.077 m/s | 0.310 m | 18/20 |

The pre-robustness checkpoint (`walk_diag_yaw/model_1600.pt`) is faster but
much more sensitive: at `cmd=0.25`, `gait_amp=0.8`, `period=0.86` it walks
0.103 m/s but survives only 8/20 perturbed starts. That version is kept for
comparison, but `walk_diag_robust/model_2100.pt` is the recommended reliable
walk.

### Earlier results

The first diagonal run (`walk_diag_strong`, 1500 iterations) produced
`model_1497.pt`, stable at 0.095 m/s over 4 s. The yaw/lateral fine-tune
(`walk_diag_yaw`, 800 iterations) produced `model_1600.pt`, stable at
0.102 m/s. Faster checkpoints reached 0.18–0.26 m/s but yawed and fell after
3–3.5 s:

| checkpoint | mean vx | distance | lateral drift | yaw | fall |
|---|---|---|---|---|---|
| 550 | 0.084 | 0.356 m | 0.097 m | −0.05 | no |
| 700 | 0.239 | 0.615 m | 0.395 m | −0.40 | 2.4 s |
| 1000 | 0.183 | 0.608 m | 0.387 m | −0.36 | 3.2 s |
| 1150 | 0.184 | 0.660 m | 0.331 m | −0.59 | 3.6 s |
| 1300 | 0.214 | 0.684 m | 0.342 m | −0.56 | 3.3 s |
| 1450 | 0.183 | 0.653 m | 0.336 m | −0.69 | 3.4 s |
| **1497** | **0.090** | **0.380 m** | **0.210 m** | **−0.15** | **no** |

The faster limit cycles still yaw and fall; the robust policy trades speed
for a wider stability basin.

The previous bound baseline (`walk5_reward`, 16 actions) is kept for reference:

| nominal bound | result |
|---|---|
| `gait_amp=0.50` | stable, 0.078 m/s, 0.316 m |
| **`gait_amp=0.55`** | **stable, 0.146 m/s, 0.656 m, body 0.632 m** |
| `gait_amp=0.60` | stable, 0.044 m/s, 0.175 m |

The nominal all-leg bound alone (zero residual) reaches ~0.25 m/s but tips
over. With the larger box and the velocity-weighted reward, PPO learns a
stable 0.15 m/s bound at `gait_amp=0.55`. Fine-tuning from `model_800` reached
0.3 m/s but the policy fell after ~2 s, so the stable 800-iteration checkpoint
is kept.

## Reward

```
3.0–4.0 × forward velocity tracking (sigma 0.15 m/s)
+ 1.0–1.5 × forward progress relative to command
+ 0.5 × hold trunk height and stay upright
− tilt, lateral-velocity, yaw-rate, and heading penalties
+ 0.3 × swing-foot clearance
− contact-aware stance-slip penalty
− tendon tension and action rate
− fall penalty
```

The slip penalty is evaluated only on stance feet that are actually near the
floor (`foot_z < 0.03 m`) and is capped:

```
r_slip = -mean(clip(|v_slip| / 0.20, 0, 2)**2)
```

This matters: the previous unbounded `(v_slip/0.05)**2` term reached −132 on
the faster nominal and dominated every other reward, which taught the policy
to creep.  The current term is about −1 to −3 on the same gait.

The swing target is `0.035 m` and the stance target is `0.015 m`.  The current
robust fine-tune uses `fall_penalty=5`, `tilt_weight=3`, `lat_weight=1.0`,
`yaw_weight=0.8`, and `heading_weight=1.0`.  There is no survival bonus, so
standing still is not rewarded once a command is present.  Falls (`z < 0.35`,
body up-vector `< 0.5`) terminate the episode.

## Notes

- The trunk is 0.60 × 0.44 × 0.10 m at 2.0 kg; the hip track is
  0.50 × 0.34 m. Four legs add ~1.24 kg each; total ~7.0 kg.
- The earlier 0.36 × 0.22 m trunk was too narrow for the leg length
  (~0.61 m): the static tip-over margin was only ~12°. The wider/longer box
  moves the support polygon out to the hips and raises that margin to ~22°.
- Feet are 15 mm spheres at the `tip` sites; leg mesh collisions are disabled.
- The servo limits are the single-leg values: 31.4 mm stroke, 200 N, with
  `kp=8000`, `kv=80` for the retrained diagonal-trot model.
- The current robust nominal is `amp=0.7`, `period=0.50`, `swing_frac=0.5`,
  `stance_amp=0.1`.  The faster `amp=1.1` / `period=0.50` / `stance_amp=0.2`
  nominal is not robust to perturbed starts.
- `rsl-rl` handles timeout bootstrapping via
  `genesis_leg.runner:LegPPO`.
