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
exports 14 bodies but no joints; `onshape_to_mjcf/merge.py --nested` groups the
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
python -m onshape_to_mjcf.build --out urdf/gradus/demo   # leg.xml + assets
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

Observations are proprioceptive (plus the commanded velocity), 115 values:

| group | count |
|---|---|
| slider position / velocity, commanded velocity, command phase (sin, cos) | 5 |
| 13 ball-joint rotation vectors | 39 |
| 13 ball-joint velocities (×0.05) | 39 |
| normalised tendon lengths | 8 |
| normalised tendon forces | 8 |
| current servo pull targets | 8 |
| previous action | 8 |

The command alternates sign every `cmd_period` seconds (2 s, ±20 % jitter).
Its magnitude is resampled in `[cmd_min, cmd_max]` at **every reversal**, so a
single fixed-amplitude pump cannot satisfy the reward; the policy has to read
the commanded magnitude.  The command phase lets it anticipate the sign flip,
but not the next magnitude.

The eight actions are residuals on the tendon pull that holds the reset pose:

```
pull = clip(pull_bias + residual_scale * action, 0, 1)
ctrl = l0 - stroke * pull
```

The servo target is low-pass filtered (50/50) and clamped to the actuator
range.  `pull_bias` is measured at reset from the bent pose, so zero action
holds the leg instead of yanking it straight.

## Reward

There are four versions, chosen with `--reward` (default `v2`).  Each one
adds to the previous one.  `--reward v1` reproduces the original reward
exactly; the `rig_v1` recipe below needs it.

**v1, the original reward:**

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

**v2 adds smoothness and hardware-damage costs.**  The command also ramps at
0.3 m/s² instead of stepping, and the action-rate weight rises from 0.02 to 0.5.

```
− 2.5 × action jerk     mean (a_t − 2 a_{t−1} + a_{t−2})²
− 0.8 × tension shock   mean (ΔT per step / 20 N)²
− 1.0 × tension limit   Σ ((T_peak / 200 N − 0.7)₊ / 0.3)²
− 1.0 × bend limit      Σ ((bend / 0.9 rad − 0.8)₊ / 0.2)²
− 0.5 × foot impact     ((F_peak − 62 N)₊ / 31 N)²
− 0.2 × foot slip       min(mean slip speed in contact / 0.05 m/s, 5)
```

These costs (and the action rate) are multiplied by a curriculum factor
k = 0.05^(0.997^iteration), which rises from 0.05 to about 0.86 at iteration
1000 (Hwangbo et al. 2019).  They can take a step's reward down to 0 but not
below, so failing early is never worth more than surviving.

**v3 frees the swing.**  The foot may lift up to 3 cm and move up to 10 cm
sideways without cost.  The hip sits 2 mm below the straight-leg contact
height, so the foot touches the floor anywhere within about 4 cm of under the
hip; the only way to bring it forward again is around the side.

**v4 scores the average speed.**  The velocity terms use the carriage speed
averaged over about one stride (exponential average, 1 s) instead of the
instantaneous speed.  One leg moves the carriage in pulses (push, glide,
swing), so its instantaneous speed swings between 0 and 0.3 m/s even when the
average is right.  Scored on instantaneous speed, the squared velocity term
ranks every walking gait below standing still.  v4 also sets the speed tolerance
(`vel_sigma`) to 0.03 m/s instead of the default 0.06, the value the v4
tests used.

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
