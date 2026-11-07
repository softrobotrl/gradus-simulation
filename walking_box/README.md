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
  rearward and lift it; the stance diagonal stays passive. Pulling front and
  back together cancels propulsion because both directions shorten the leg.
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

Commands are kept strictly positive during walking so the policy cannot settle
into a stable standing local optimum.

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

## Evaluate / render

```bash
python -m walking_box.evaluate \
    --ckpt logs/walking_box/walk_diag_yaw/model_1600.pt \
    --cmd 0.25 --gait-amp 0.7 --period 0.8 --steps 400
```

Writes `eval/metrics.txt` and `eval/walk.gif` (3/4 view, 25 fps at
`--every 4`). The camera follows the trunk so the gait stays framed. Add
`--no-render` to write metrics only.

### Result (walk_diag_strong, 1500 iterations)

Best stable checkpoint: `model_1497.pt`, evaluated at `cmd=0.25 m/s`,
`gait_amp=0.7`, `period=0.8`, 400 steps (4 s):

| metric | result |
|---|---|
| mean forward speed | 0.095 m/s |
| distance | 0.380 m |
| lateral drift | 0.210 m |
| yaw | −0.15 rad |
| trunk height | 0.628 m |
| fall | no |

Earlier checkpoints reached 0.18–0.26 m/s but yawed and fell after 3–3.5 s
(`model_700`, `model_1000`, `model_1150`, `model_1300`, `model_1450`):

| checkpoint | mean vx | distance | lateral drift | yaw | fall |
|---|---|---|---|---|---|
| 550 | 0.084 | 0.356 m | 0.097 m | −0.05 | no |
| 700 | 0.239 | 0.615 m | 0.395 m | −0.40 | 2.4 s |
| 1000 | 0.183 | 0.608 m | 0.387 m | −0.36 | 3.2 s |
| 1150 | 0.184 | 0.660 m | 0.331 m | −0.59 | 3.6 s |
| 1300 | 0.214 | 0.684 m | 0.342 m | −0.56 | 3.3 s |
| 1450 | 0.183 | 0.653 m | 0.336 m | −0.69 | 3.4 s |
| **1497** | **0.090** | **0.380 m** | **0.210 m** | **−0.15** | **no** |

A yaw/lateral fine-tune (`walk_diag_yaw`) from `model_1497` was run for 800
iterations with `lat_weight=2.0`, `yaw_weight=1.5`, and `heading_weight=1.0`.
The furthest stable checkpoint is `model_1600.pt`:

| metric | result |
|---|---|
| mean forward speed | 0.102 m/s |
| distance | 0.406 m |
| lateral drift | 0.237 m |
| yaw | −0.17 rad |
| trunk height | 0.627 m |
| fall | no |

Checkpoints `1650` and `1700` reach 0.19–0.20 m/s but yaw and fall after
3.7–3.9 s. The final fine-tune checkpoint (`2296`) is stable but back to
0.089 m/s. The fast limit cycle is not yet robustly stable.

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
2.0 × forward velocity tracking (sigma 0.10 m/s)
+ 0.7 × forward progress relative to command
+ 0.5 × hold trunk height and stay upright
− lateral-velocity penalty
− yaw-rate and heading penalties
+ 0.3 × swing-foot clearance
− stance-foot slip penalty
− tendon tension and action rate
− 1.0 on a fall
```

The swing target is `0.035 m` and the stance target is `0.015 m`. The
yaw/lateral fine-tune raises `lat_weight` to 2.0, `yaw_weight` to 1.5, and
`heading_weight` to 1.0. There is no survival bonus, so standing still is not
rewarded once a command is present. Falls (`z < 0.35`, body up-vector `< 0.5`)
terminate the episode.

## Notes

- The trunk is 0.60 × 0.44 × 0.10 m at 2.0 kg; the hip track is
  0.50 × 0.34 m. Four legs add ~1.24 kg each; total ~7.0 kg.
- The earlier 0.36 × 0.22 m trunk was too narrow for the leg length
  (~0.61 m): the static tip-over margin was only ~12°. The wider/longer box
  moves the support polygon out to the hips and raises that margin to ~22°.
- Feet are 15 mm spheres at the `tip` sites; leg mesh collisions are disabled.
- The servo limits are the single-leg values: 31.4 mm stroke, 200 N, with
  `kp=8000`, `kv=80` for the retrained diagonal-trot model.
- `rsl-rl` handles timeout bootstrapping via
  `genesis_leg.runner:LegPPO`.
