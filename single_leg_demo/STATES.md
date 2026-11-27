# Single-leg rig: policy states (hardware-measurable)

Status: **implemented** in `rig_env.py` (`obs_mode="hardware"`, the default).

**Hardware update (2026-11-25):** the servos are PWM servos with AS5600 magnetic encoders
(4096 counts per revolution) read over a TCA9548A I2C mux.  PWM servos report **no load**, so
row 6 is dropped by default (`servo_load_obs=False`): **33 values per step, 33 × 8 = 264**.
Encoder positions are rounded to whole counts (`encoder_counts_per_rev`, `spool_radius`).
Runs from before this change used the 41-value table below and still load as they were.
**Update 2026-11-26:** the rig has **no IMU**, so rows 7-8 are dropped by default
(`imu_obs=False`), and the spool radius is 13.5 mm (`spool_radius`; 148 N per cable at the
assumed 4 N·m).  With 4 servos the policy input is **15 values per step, 15 × 8 = 120**:
command and phase (3), previous action (4), servo targets (4), encoder positions (4).
**Done since:** the leg has 4 servos, each a double spool driving 2 tendons; the sim still has 8
servos until the tendon pairing is confirmed.

## The state, per control step (every 20 ms)

| # | name | count | what it is | units / scaling | where it comes from on hardware |
|---|---|---|---|---|---|
| 1 | target velocity | 1 | the carriage speed we are asking for (+ = move +X, − = move −X) | m/s ÷ 0.12 | set by our controller (not measured) |
| 2 | phase (sin, cos) | 2 | how far we are into the current back-and-forth segment | sin/cos of 2π·(t / period) | computed from a clock (not measured) |
| 3 | previous action | 8 | the raw policy output from the previous step | −1..1 | the policy's own last output (not measured) |
| 4 | last servo targets | 8 | the position each servo was **told** to go to in the previous step | 0 = spool slack, 1 = full 31.4 mm pull | the value our controller sent (not measured) |
| 5 | servo positions | 8 | where each servo spool **actually is** now | same 0..1 scale as #4, in whole encoder counts | AS5600 encoders |
| 6 | servo loads | 8 | how hard each servo is pulling on its tendon (**only with `servo_load_obs`**; PWM servos have no load reading) | fraction of max servo force (0..1.5) | servo load reading |
| 7 | rail IMU accelerometer | 3 | acceleration of the carriage (x, y, z) | g (reads ≈ 0, 0, 1 when still) | IMU on the rail carriage |
| 8 | rail IMU gyro | 3 | angular rate of the carriage (x, y, z) | rad/s ÷ 4 (≈ 0 on a straight rail) | IMU on the rail carriage |
| | **total** | **33** (41 with loads) | | | |

The values are stored in this order (rows 1 → 8) inside each step.

### History

The policy sees the **last 8 steps** of this table stacked together, oldest first, so its input is
**33 × 8 = 264 values** (0.16 s; 328 with loads). History stacks the whole table, so it contains the last 8 sensor
readings **and**, through row 3, the last 8 policy outputs. After a reset, all 8 slots are filled with
the first step. Change the length with `--obs-history`.

## Output

8 values, one per tendon servo, in [−1, 1]. Each becomes the next servo target:

```
new target  = pull_bias + 0.6 × output                        (clipped to 0..1)
sent target = 0.5 × previous sent target + 0.5 × new target   (smoothing)
```

`pull_bias` is fixed: every tendon just taut with the leg straight (pull 0 for this model), a one-off
calibration on hardware. The servo then moves toward the sent target at its top speed
(`servo_speed`, assumed 0.07 m/s at the spool), so it can lag behind the target.

## Time order: what is input and what is output

Nothing is an input and an output at the same moment. The output of one step comes back as an
input to the **next** step:

```
step t-1:  state(t-1) ──► policy ──► output(t-1) ──► servo target(t-1) sent to servos
                                                            │
                               servos move for 20 ms        │
                                                            ▼
step t:    sensors now + output(t-1) + target(t-1) ──► policy ──► output(t) ──► target(t)
```

Three different servo-related quantities appear in the state:

| | what it is | when |
|---|---|---|
| previous action (#3) | raw policy output, −1..1 | output at step t-1 |
| last servo target (#4) | that output converted to a servo position (+ `pull_bias`, smoothed): what was **sent** | sent at step t-1 |
| servo position (#5) | where the servo **actually is** | measured at step t |

#3 and #4 overlap (one is computed from the other). Both are kept on purpose; dropping #3 would
lose little. #4 vs #5 is the useful comparison: a servo that stopped short of its target is held
back by a loaded tendon.  Without loads (PWM servos), that gap is the policy's only direct sign of
cable tension.

## Removed from the old 115-value state

| removed | count | why |
|---|---|---|
| ball-joint rotations | 39 | no sensor on the joints |
| ball-joint velocities | 39 | no sensor on the joints |
| slider (carriage) position | 1 | no rail encoder |
| slider (carriage) velocity | 1 | no rail encoder |
| tendon lengths | 8 | replaced by servo positions (#5), which is what the encoders measure |
| tendon forces | 8 | replaced by servo loads (#6), which is what the servo reports |

## What still uses simulator-only information (and why that is allowed)

These are used **only during training** and never run on the robot:

- **Critic (value function):** gets the full old 115-value state, the carriage speed the reward scores
  (the ~1 s average under reward v4) and 13 domain-randomized values
  (friction, rail height, battery, delay, …), 129 in total (observation group `critic`), including the true
  carriage position and velocity. It only exists to train the policy.
- **Reward and episode ending:** use the true carriage velocity (to score tracking) and the true
  carriage position (to penalise / end episodes near the rail ends).

On hardware there is no rail encoder, so nothing stops the carriage at the rail ends except the
policy's training. Use end stops or limit switches as a safety backstop.

`obs_mode="full"` (`--obs-mode full`) gives the policy the 115-value state again, for comparison runs
and old checkpoints. `evaluate.py` reads the mode from each run's `config.json`.

## Simulator mapping

| state | simulator source (`rig_env.py`, `_hw_frame`) |
|---|---|
| target velocity | `env.cmd[i] / obs_cmd_scale` (`cmd` in code = target velocity) |
| phase | `env._phase(i)` |
| previous action | `env.prev_action[i]` |
| last servo targets | `(l0 - env.servo_cmd[i]) / stroke` (what was sent) |
| servo positions | `min(tendon pull, setpoint)`: the setpoint `d.ctrl` chases the sent target at `servo_speed`; a taut tendon holds the spool short of it, a slack tendon lets it get there; tendon pull = `(l0 - d.ten_length) / stroke` |
| servo loads | `-d.actuator_force / fmax` |
| rail IMU | MuJoCo `accelerometer` (`imu_acc`) and `gyro` (`imu_gyro`) on the `imu` site on top of the carriage, added by `build_rig.py` |

## Open items

- **Sensor noise:** with `--dr`, servo positions, servo loads and the IMU get noise and bias
  (assumed MPU6050-class IMU); see `domain_randomization.md`. Target velocity, phase, previous
  action and last servo targets stay exact, since the controller knows them.
- **Servo speed:** `servo_speed` is an assumed hobby-servo value; replace it with the datasheet
  speed once the servo is known.
- **Load scaling** (only if a servo with a load reading is used again): calibrate it to the same
  0..1 fraction of max force used here.
