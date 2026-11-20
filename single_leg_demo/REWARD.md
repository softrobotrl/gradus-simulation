# Changing the reward

For anyone, person or coding agent, who changes or adds reward terms for the
single-leg rig.  Read this before editing; it takes five minutes.

## How the reward is built

Every control step (20 ms), for each env:

```
reward = Σ over terms  weight × k × term(state)
         (bounded costs capped so the step stays ≥ 0)
         − failure_penalty   if the episode failed
```

- **term(state)** is a small function in [rewards.py](rewards.py) that returns a
  raw value: positive for something good, negative for a cost.  `state` holds
  everything a term may read (carriage speed, foot height, tensions, actions...).
- **weight** is a field of `RigConfig` ([rig_env.py](rig_env.py)), so every run's
  `config.json` records it.
- **k** is the cost curriculum: it multiplies the smoothness and damage costs,
  rising from 0.05 to about 0.86 over 1000 iterations, so the policy learns to
  move before it learns to move gently.
- **Bounded costs** (`bound_costs`): the smoothness and damage costs may pull a
  step down to 0 but not below, so failing early never pays better than surviving.

The reward runs only in training, so it may use simulator-only quantities (true
carriage speed, foot contact force) that the real robot cannot measure.

**Presets** `v1` to `v4` are named sets of weights (`PRESETS` in rewards.py),
chosen with `train.py --reward`.  The default is `v4`.

## The current reward (v4)

| Term | Rewards or penalises | Weight field | v4 value |
|---|---|---|---|
| `vel` | speed tracking, 0..1 bell curve of (scored speed − command), width `vel_sigma` 0.03 m/s | `vel_weight` | 2.0 |
| `vel_l2` | squared speed error (keeps a gradient far from the target) | `vel_l2_weight` | 0.02 |
| `prog` | moving in the commanded direction (speed ÷ command, 0..1.5) | `progress_weight` | 0.5 |
| `rail` | getting within 30 % of a rail end | `rail_weight` | 0.5 |
| `foot` | foot lift beyond `foot_free` 3 cm | `foot_weight` | 0.3 |
| `lat` | foot sideways beyond `lat_free` 10 cm | `lat_weight` | 0.5 |
| `effort` | squared tendon pull | `effort_weight` | 0.02 |
| `ten` | tendon tension | `tension_weight` | 0.001 |
| `action_l2` | raw action size (stops bang-bang saturation) | `action_l2_weight` | 0.005 |
| `rate` * | action change per step | `rate_weight` | 0.5 |
| `accel` * | jerk: sudden reversals of the servo targets | `accel_weight` | 2.5 |
| `shock` * | sudden tension jumps | `shock_weight` | 0.8 |
| `tlim` * | cables above 70 % of max tension | `tension_limit_weight` | 1.0 |
| `blim` * | joints above 80 % of the bend limit | `bend_limit_weight` | 1.0 |
| `impact` * | foot force above 2× the standing load | `impact_weight` | 0.5 |
| `slip` * | foot sliding while planted | `slip_weight` | 0.2 |

\* scaled by the curriculum k and bounded.  Failure (foot above 12 cm, a joint
bent past 0.9 rad, or the carriage at a rail end) costs `failure_penalty` 5.0
and ends the episode.

The **scored speed** is the carriage speed averaged over about one stride
(`vel_avg_tau` 1 s).  This is the key v4 change: one leg moves the carriage in
pulses, and scoring the instantaneous speed ranked every gait below standing still.

TensorBoard logs each term's weighted value as `/r_<name>`; they sum to
`/reward_mean` before the bounded-cost cap and the failure penalty.  `/clipped`
is the share of steps where the cap applied.

## Try a change without editing code

Any `RigConfig` field can be set from the command line:

```
python -m single_leg_demo.train --run-name my_test --set accel_weight=1.0 --set lat_free=0.15
```

A misspelt field is an error, not a silent no-op.  Values are parsed by the
field's type (`true`/`false`, numbers, `(0.3, 1.0)` for ranges).

## Add a term

1. Write the function in [rewards.py](rewards.py): `def my_term(s, c) -> float`,
   positive for good, negative for a cost.  `s` is the `StepState`, `c` the config dict.
2. Add `Term("my_term", my_term, "my_term_weight")` to `TERMS` (add
   `curriculum=True, bounded=True` for a smoothness or damage cost).
3. Add `my_term_weight: float = 0.0` to `RigConfig` in [rig_env.py](rig_env.py).
   Default 0 keeps every existing preset unchanged.
4. If the term needs a quantity `StepState` lacks, add the field there and fill
   it in `SingleLegRigEnv._reward_one`.  If it depends on history the policy
   cannot see (like an average), consider giving it to the critic as well
   (`_obs_critic`), as v4 does with the scored speed.
5. Add its parameter names to `REWARD_FIELDS` in rewards.py if they only
   change the reward (lockstep scoring relies on this list).
6. To make it a preset, add `"v5": {...}` to `PRESETS`.  Do not edit old presets:
   past runs and results refer to them.

## Check before you train

A training run takes about 30 minutes; these take seconds.

```
python -m single_leg_demo.reward_check.preference --set my_term_weight=0.5
python -m single_leg_demo.reward_check.preference --compare v1 v4
python -m single_leg_demo.reward_check.lockstep --a v4 --b v4 --set-b my_term_weight=0.5 \
    logs/single_leg_demo/RUN/model_998.pt ...
python -m unittest tests.test_single_leg_demo
```

- **Preference test** (about 3 s): scores a scripted walk, a jerky walk, a noisy
  walk and standing still.  The walk must beat standing (else FAIL: training
  will find standing), and should beat the jerky and noisy walks (else WARN).
- **Lockstep scoring:** scores trained checkpoints under two rewards on exactly
  the same motion, plus reward-free measures (distance covered / asked, foot
  lifts).  A good reward ranks policies by how well they do the task.
- The unit tests include the preference test: v4 must pass and v1 must fail.

## Lessons so far

- **Passing the preference test is necessary, not sufficient.**  As of
  2026-11-19, PPO trained from scratch has not learned to walk under any preset
  (v4 covered about 10 % of the asked distance at 0.2 N rail friction and 0 % at
  3 N).  Turning off the rate, jerk and shock costs did not change that.
  Likely causes, both unverified: walking needs a coordinated
  plant-push-swing that random exploration rarely finds, and runs are short.
- **v4 rewards covering distance, not steps.**  A policy can score well by
  leaning or shuffling.  A term that asks for foot lifts is an open idea.
- **Costs on action changes also charge PPO's exploration noise** (action std
  about 0.5).  Watch `/r_accel` and `/clipped` early in a run: if the costs
  grow with k while the policy is not changing, they are mostly pricing noise.
- **Do not punish the side swing.**  The foot touches the floor anywhere within
  about 4 cm of under the hip, so a step has to swing around the side (why
  `lat_free` and `foot_free` exist).
