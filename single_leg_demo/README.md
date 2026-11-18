# Single demo leg on a sliding rig

One [Demo-Leg](https://cad.onshape.com/documents/ee938c509aab09f13ed20423/w/91dfacf08bedc9f37dd5152e/e/cf7c030f13cd05e6297e246f)
on a one-dimensional linear rail.  The hip is bolted to a carriage; the only
ground contact is a small sphere at the foot.  PPO learns to bend the leg with
its eight tendon servos so that the carriage tracks a commanded velocity and
pumps itself back and forth.

```
              rail  ─────────────────────────────►  +X
                       [carriage]
                           │ hip (fixed to carriage)
                         ╭─╯
                        ╱   thigh (6 disks)
                       │
                        ╲   shin (6 disks)
                         ╰─ foot sphere on the floor
```

## CAD and tendon routing

| Item | ID |
|---|---|
| Document | `ee938c509aab09f13ed20423` (`Demo-Leg`) |
| Workspace | `91dfacf08bedc9f37dd5152e` |
| Assembly | `cf7c030f13cd05e6297e246f` (`Sim-Leg-Assembly`) |

The assembly has 14 stations (hip mount + 6 thigh segments + 6 shin segments +
foot adaptor).  Each station is a sub-assembly containing the two printed
halves, and the top level has 13 ball mates.  `onshape-to-robot` therefore
exports 14 bodies but no joints; `mujoco_to_mjcf/merge.py --nested` groups the
bodies by occurrence, nests them into a chain, and adds the ball joints.

Tendon routing is the same scheme as the original leg:

- **4 inner tendons** at r = 19.1 mm, hip → foot (14 sites each),
- **4 outer tendons** at r = 29.1 mm, hip → last thigh disk (7 sites each),

for 8 spatial tendons and 8 position-controlled tendon servos (2 N·m,
10 mm spool, 31.4 mm stroke, kp = 8000, kv = 80).  The demo disk holes are
larger than the old 2.1 mm holes and the hip plate is clocked 45° relative to
the disk stacks; the merge detects the holes by their radial distance from the
backbone bore and keeps the old direction-naming rule.

## Build

```bash
set -a && . ./.env && set +a
python -m mujoco_to_mjcf.build --out urdf/gradus/demo   # leg.xml + assets
python -m single_leg_demo.build_rig                     # rig.xml
python -m single_leg_demo.validate --num-envs 4
```

`urdf/gradus/demo/leg.xml` is the 14-link tendon leg with the hip fixed at the
origin.  `urdf/gradus/demo/rig.xml` adds the carriage, the prismatic `slide`
joint along +X, the floor, and a 15 mm foot sphere.  The rail height is derived
from the generated leg (straight hip-to-foot length minus 2 mm), so the leg
starts slightly compressed; the environment additionally seeds a small bend at
every reset.

## Observations and actions

The policy only sees what the real rig can know. Full definitions, units and
time order are in [STATES.md](STATES.md). Per 20 ms step, 41 values:

| group | count | source on hardware |
|---|---|---|
| target velocity, phase (sin, cos) | 3 | our controller / clock |
| previous action | 8 | last policy output |
| last servo targets | 8 | what our controller sent |
| servo positions | 8 | servo encoders |
| servo loads | 8 | servo load reading |
| rail IMU accelerometer + gyro | 6 | IMU on the carriage |

The last 8 steps are stacked, so the policy input is 328 values.  The critic
(training only) gets the full 115-value simulator state: slider position and
velocity, target velocity, phase, 13 ball-joint rotations and velocities,
tendon lengths and forces, servo targets and the previous action, plus 13
domain-randomized values (128 in total).  Use `--obs-mode full` to give the
policy the 115-value state instead (old behaviour).  Domain randomization
(`--dr`) is described in [domain_randomization.md](domain_randomization.md).

The target velocity ("command", `cmd` in code) alternates sign every `cmd_period` seconds (2 s, ±20 % jitter).
Its magnitude is resampled in `[cmd_min, cmd_max]` at **every reversal**, so a
single fixed-amplitude pump cannot satisfy the reward; the policy has to read
the commanded magnitude.  The command phase lets it anticipate the sign flip,
but not the next magnitude.

The eight actions are residuals on a fixed tendon pull:

```
pull = clip(pull_bias + residual_scale * action, 0, 1)
ctrl = l0 - stroke * pull
```

`pull_bias` is a one-off calibration: every tendon just taut with the leg
straight, which is pull 0 for this model.  The sent servo target is low-pass
filtered (50/50) and clamped to the actuator range.  The simulated servo then
moves toward it at no more than `servo_speed` (0.07 m/s at the spool, an
assumed hobby-servo 0.15 s / 60°, until the real servo is known).

## Reward

```
2.0 × velocity tracking          exp(-0.5 ((vx - cmd) / 0.05)^2)
− 0.02 × squared velocity error  keeps a gradient through reversals
+ 0.5 × progress in command dir  clip(vx / cmd, 0, 1.5)
− 0.5 × rail-end soft barrier
− 0.3 × foot height error        keep the foot on the floor
− 0.5 × lateral foot excursion   keep the bend in the X-Z plane
− effort, action rate, tension
− 0.01 × raw action²          keep the policy out of hard saturation
− 5.0 on failure
```

Episodes end on a foot lift above 0.12 m, a ball-joint bend above 0.9 rad, a
carriage position beyond 98 % of the rail, or NaN.  Timeouts bootstrap values
through `genesis_leg.runner:LegPPO`.

## Train

Short smoke run:

```bash
python -m single_leg_demo.train --run-name rig_smoke \
    --num-envs 16 --iterations 5 --steps-per-env 8 --device cpu
```

Full run:

```bash
python -m single_leg_demo.train --run-name rig_v1 \
    --num-envs 64 --iterations 1000 --steps-per-env 24 \
    --num-threads 8 --episode-steps 400 --warmup 20 \
    --cmd-min 0.06 --cmd-max 0.14 --vel-sigma 0.05 --device cuda
```

The first 20 iterations use a reduced command (0.05–0.08 m/s); the rest use
0.06–0.14 m/s.  The command magnitude changes at every reversal.  Checkpoints
go to `logs/single_leg_demo/<run>/model_*.pt`.
CUDA trains the small MLP about four times faster than CPU; the environment
itself always runs on the CPU.

Fine-tune or resume:

```bash
python -m single_leg_demo.train \
    --resume logs/single_leg_demo/rig_v1/model_998.pt \
    --iterations 500 --num-envs 64 --num-threads 8 \
    --run-name rig_v2 --cmd-min 0.04 --cmd-max 0.16 \
    --vel-l2-weight 0.05 --vel-sigma 0.05 \
    --learning-rate 2e-4 --warmup 0 --device cuda
```

`cmd_resample` is on by default: the commanded magnitude is redrawn at every
reversal.  This is what breaks the fixed-amplitude pump that an earlier
per-episode command produced.

Useful overrides: `--residual-scale`, `--cmd-period`, `--vel-sigma`,
`--vel-l2-weight`, `--action-l2-weight`, `--reset-noise`, `--bend-limit`,
`--entropy-coef`, `--learning-rate`, `--std`.

### Result (`rig_v3`, 400 fine-tune iterations)

The policy was trained in three stages: 1000 iterations with a per-episode
command (`rig_v1`), a 500-iteration fine-tune that resamples the command
magnitude at every reversal (`rig_v2`), then a 400-iteration fine-tune that
adds the raw-action penalty (`rig_v3`).  Best checkpoint:
`rig_v3/model_1896.pt`.

Evaluated for 400 control steps (8 s) at a fixed command, 2 s per direction:

| command | mean \|err\| | RMSE | carriage span | contacts | fall |
|---|---|---|---|---|---|
| 0.04 m/s | 0.013 | 0.018 | 0.113 m | 100 % | no |
| 0.08 m/s | 0.017 | 0.028 | 0.205 m | 100 % | no |
| **0.10 m/s** | **0.019** | **0.033** | **0.240 m** | **100 %** | **no** |
| 0.12 m/s | 0.022 | 0.043 | 0.265 m | 100 % | no |
| 0.16 m/s | 0.045 | 0.075 | 0.340 m | 100 % | no |

The carriage span scales with the command, so the policy really reads the
commanded velocity rather than replaying a fixed pump.  All runs complete four
reversals with the foot on the floor.

## Evaluate

```bash
python -m single_leg_demo.evaluate \
    --ckpt logs/single_leg_demo/rig_v3/model_1896.pt \
    --steps 400 --cmd-min 0.10 --cmd-max 0.10 --cmd-period 2.0
```

Writes `eval/metrics.txt` and `eval/rig.gif`.  The metrics are mean absolute
velocity error, RMSE, carriage span, foot-contact fraction, maximum ball-joint
bend, and the number of command reversals completed.  Add `--no-render` for
metrics only, or run the GIF render under `xvfb-run -a env MUJOCO_GL=glfw` on a
headless machine.

## Notes

- Moving mass is about 3.65 kg (1.63 kg leg + 2.0 kg carriage).  The rail has
  `damping=0.6` and `frictionloss=0.2`; the foot friction is 1.0.
- Leg mesh collisions are disabled, so the lower disks can fold without
  fighting the floor.  Only the 15 mm foot sphere contacts.
- The backbone stiffness is the same rod model as the other legs:
  7.94 mm, 30 GPa, 45.7 mm pitch → 128 N·m/rad per ball joint.
- The policy is a 256×256 MLP actor/critic with PPO (4 epochs, 4 minibatches,
  lr 3e-4, adaptive KL, 24 steps/env per iteration).
- The first long run drifted into bang-bang actions (`|a| ≈ 8`).  Clipping to
  `[-1, 1]` then erased the commanded magnitude, and the leg pumped at a fixed
  amplitude.  Resampling the command every reversal plus the raw-action
  penalty brought the RMS action down to ≈2 and restored velocity scaling.
- The command observation is normalised by a fixed `obs_cmd_scale` (saved in
  `config.json`), not by the per-run `cmd_max`.  Without that, evaluating the
  same policy at different commands looks identical because every command maps
  to the same normalised value.
