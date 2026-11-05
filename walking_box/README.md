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

1. 300 iterations with the nominal gait off, `cmd_max=0` (stand)
2. 400 at `cmd_max=0.10` m/s with the nominal bound on
3. 800 at `cmd_max=0.25` m/s

The policy is a 256×256 MLP actor/critic; PPO with 4 epochs, 4 minibatches,
`lr=3e-4`, adaptive KL. Checkpoints go to
`logs/walking_box/<run>/model_*.pt`.

## Evaluate / render

```bash
python -m walking_box.evaluate \
    --ckpt logs/walking_box/walk1/model_1500.pt \
    --cmd 0.2 --steps 400
```

Writes `eval/metrics.txt` and `eval/walk.gif` (3/4 view, 25 fps).

## Reward

```
track forward body velocity
+ hold trunk height and stay upright
− tendon tension, action rate, yaw rate
+ small survival bonus
```

Falls (`z < 0.35`, body up-vector `< 0.5`) terminate the episode.

## Notes

- The trunk is 1.5 kg; four legs ~1.24 kg each; total ~6.5 kg.
- Feet are 15 mm spheres at the `tip` sites; leg mesh collisions are disabled.
- The servo limits are the single-leg values: 31.4 mm stroke, 200 N.
- `rsl-rl` handles timeout bootstrapping via
  `genesis_leg.runner:LegPPO`.
