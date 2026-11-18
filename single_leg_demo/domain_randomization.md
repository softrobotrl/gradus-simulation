# Single-leg rig: domain randomization

Status: steps 1–2 **done**; steps 3–7 planned.

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
- **The critic also sees the randomized values** (friction, battery level, …), since it never runs
  on hardware. This makes training easier.
- **Ramp-up:** training starts with no randomization and widens to the full ranges over the first
  few hundred iterations.

## Implementation order

1. ✅ Servo speed limit: `servo_speed`, assumed 0.07 m/s at the spool.
2. ✅ Fixed `pull_bias`: tendons just taut with the leg straight (pull 0 for this model).
3. One MuJoCo model per env, randomized at reset; `dr_enabled` switch.
4. Voltage sag, control delay and pushes in the step loop; sensor noise in `_hw_frame`.
5. Randomized values added to the critic input; ramp-up in `train.py`.
6. Robustness sweep in `evaluate.py`: tracking error vs friction, battery level and delay.
7. Train with and without randomization and compare on the sweep.

## Assumptions

- **Power source:** battery, so voltage sag is included at the ranges above.
- **Servo model:** unknown. The speed limit and force-vs-voltage use generic hobby-servo values,
  kept in the config so a datasheet can replace them later.
- **No measurements:** rail friction, carriage mass, foot friction etc. will not be measured; the
  wide ranges cover that uncertainty.
