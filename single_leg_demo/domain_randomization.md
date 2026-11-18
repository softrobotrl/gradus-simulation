# Single-leg rig: domain randomization

Status: steps 1–6 **done**; step 7 planned.

Turn it on with `python -m single_leg_demo.train --dr` (`RigConfig(dr_enabled=True)`); it is off by
default. `python -m single_leg_demo.validate --dr` prints each env's drawn values.

## Why

Without randomization the policy trains on one exact set of physics and learns to rely on it.
Randomizing the physics every episode forces it to work across a range; the real rig is then just
one more variation it has already seen.

## What is randomized

Drawn once per episode. ×1.0 = current value.

| group | parameter | current value | range |
|---|---|---|---|
| **contact** | foot–floor friction | 1.0 | 0.5 – 1.3 |
| **rail** | carriage friction | 0.2 N | ×0.5 – 2.0 |
| | carriage damping | 0.6 | ×0.5 – 2.0 |
| | carriage mass | 2.0 kg | ×0.8 – 1.2 |
| **leg** | segment masses | from CAD | ×0.9 – 1.1 each |
| | backbone stiffness | 128 | ×0.8 – 1.2 |
| | joint damping | 0.5 | ×0.7 – 1.5 |
| **servos** | position gain (stiffness) | 8000 | ×0.7 – 1.3 |
| | strength per servo | 200 N | ×0.9 – 1.1 each |
| | zero calibration per servo | — | ±1 mm |
| **voltage sag** | battery level | 1.0 | 0.8 – 1.0 |
| | sag under load | none | 0 – 25 % at full load |
| **timing** | control delay | 0 ms | 0 – 20 ms |
| **sensors** | IMU noise and bias | none | MPU6050-class |
| | servo position noise | none | encoder resolution |
| | servo load noise and scale | none | ±20 % scale, plus noise |
| **disturbance** | random pushes on the carriage | none | occasional small push |

The rig's real values are not measured, so the ranges are deliberately wide to make the policy
robust rather than to match the hardware. They live in `RigConfig`.

Sensor noise values (policy input only):

| sensor | noise |
|---|---|
| IMU accelerometer | bias ±0.03 g per episode + noise σ 0.01 g |
| IMU gyro | bias ±0.02 rad/s per episode + noise σ 0.01 rad/s |
| servo position | rounded to 1/2048 of the stroke + noise σ 0.002 |
| servo load | × 0.8–1.2 per servo per episode + noise σ 0.03 |

## Voltage sag

Lower voltage makes a servo weaker and slower. It drops more when all servos pull hard at once, so
it is recomputed every step:

```
voltage   = battery_level × (1 − sag × average servo load)
max force = 200 N × voltage
max speed = servo speed × voltage
```

This needs a servo speed limit, which the sim does not have yet (see step 1 below).

## Training only

- **Sensor noise** is added to the policy's input only; the critic sees clean values.
- **The critic also sees the randomized values**: 12 extra inputs (friction, rail friction and
  damping, carriage mass, stiffness, joint damping, servo gain and strength, battery, sag, delay,
  current voltage), since it never runs on hardware. This makes training easier.
- **Ramp-up:** training starts with no randomization and widens linearly to the full ranges over
  `--dr-ramp` iterations (default 300), updated every 50 iterations. A resumed run uses full ranges.

## Implementation order

1. ✅ Servo speed limit: `servo_speed`, assumed 0.07 m/s at the spool.
2. ✅ Fixed `pull_bias`: tendons just taut with the leg straight (pull 0 for this model).
3. ✅ One MuJoCo model per env (4 MB each), randomized at reset; `dr_enabled` switch.
   `dr_override` fixes any parameter by name, e.g. `{"friction": 0.4}`.
4. ✅ Voltage sag, control delay and pushes in the step loop; sensor noise in `_hw_frame`.
   About 25 % slower simulation with randomization on.
5. ✅ Randomized values added to the critic input (127 values); ramp-up in `train.py` (`--dr-ramp`).
6. ✅ Robustness sweep: `python -m single_leg_demo.robustness --ckpt <model.pt>`. Changes one
   parameter at a time (friction, rail friction, carriage mass, battery, sag, delay up to 30 ms,
   servo gain), then noise + pushes, then everything randomized; some values go past the training
   ranges. Prints tracking error, failure rate and survival time, and writes a CSV next to the
   checkpoint.
7. Train with and without randomization and compare on the sweep.

## Assumptions

- **Power source:** battery, so voltage sag is included at the ranges above.
- **Servo model:** unknown. The speed limit and force-vs-voltage use generic hobby-servo values,
  kept in the config so a datasheet can replace them later.
- **No measurements:** rail friction, carriage mass, foot friction etc. will not be measured; the
  wide ranges cover that uncertainty.
