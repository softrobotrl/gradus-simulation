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
- **Nominal all-leg bound**: a phase-based pattern pulls all four legs' front
  tendons during the recovery half-cycle and back tendons during the push
  half-cycle. The continuum legs cannot lift their feet above straight, so a
  bound/shuffle is more realistic than a swing-based trot.
- **16 policy actions**: residual tendon pulls for group A (FL + BR) and group
  B (FR + BL). `pull = nominal + 0.5 * residual`, clipped to `[0, 1]`. The
  diagonal pairs are tied, so the policy has half the output dimension.
- 104 observations: trunk quat, body velocity, angular velocity, height error,
  command, gait phase, 32 tendon lengths, 32 tendon forces, 4 foot heights,
  4 foot vertical velocities, previous action.
- CPU MuJoCo + `rsl-rl` PPO; no MJX/Warp required.

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

1. 200 iterations with the nominal gait off, `cmd_max=0` (stand)
2. 300 at `cmd=0.05..0.15` m/s with the nominal bound on
3. 1000 at `cmd=0.10..0.25` m/s

Commands are kept strictly positive during walking so the policy cannot settle
into a stable standing local optimum.

The policy is a 256×256 MLP actor/critic; PPO with 4 epochs, 4 minibatches,
`lr=3e-4`, adaptive KL. Checkpoints go to
`logs/walking_box/<run>/model_*.pt`. Torch and NumPy are seeded from
`EnvConfig.seed`, so repeated runs are reproducible.

To fine-tune a checkpoint without replaying the stand/walk curriculum:

```bash
python -m walking_box.train_rsl \
    --resume logs/walking_box/walk5_reward/model_800.pt \
    --iterations 500 --num-envs 64 --steps-per-env 24 \
    --num-threads 8 --episode-steps 400
```

## Evaluate / render

```bash
python -m walking_box.evaluate \
    --ckpt logs/walking_box/walk5_reward/model_800.pt \
    --cmd 0.25 --gait-amp 0.55 --steps 400
```

Writes `eval/metrics.txt` and `eval/walk.gif` (3/4 view, 25 fps). The camera
follows the trunk so the gait stays framed.

### Result (walk5_reward, 900 iterations)

Best checkpoint: `logs/walking_box/walk5_reward/model_800.pt`, evaluated at
`cmd=0.25 m/s` for 400 steps (4 s):

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
2.0 × track forward body velocity
+ 0.5 × hold trunk height and stay upright
− tendon tension, action rate, yaw rate
+ 0.02 survival bonus
```

Falls (`z < 0.35`, body up-vector `< 0.5`) terminate the episode.

## Notes

- The trunk is 0.60 × 0.44 × 0.10 m at 2.0 kg; the hip track is
  0.50 × 0.34 m. Four legs add ~1.24 kg each; total ~7.0 kg.
- The earlier 0.36 × 0.22 m trunk was too narrow for the leg length
  (~0.61 m): the static tip-over margin was only ~12°. The wider/longer box
  moves the support polygon out to the hips and raises that margin to ~22°.
- Feet are 15 mm spheres at the `tip` sites; leg mesh collisions are disabled.
- The servo limits are the single-leg values: 31.4 mm stroke, 200 N.
- `rsl-rl` handles timeout bootstrapping via
  `genesis_leg.runner:LegPPO`.
