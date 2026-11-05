# Tendon-leg tracer

PPO policy that makes the **tip-up** segmented tendon leg trace arbitrary
shapes. The base is fixed; the tip's **Z is free**, and only the **XY**
projection is rewarded, so the controller can raise/lower the foot to reach a
point.

```
                 fixed base (z=0)
                      │
                 ╭────┴────╮
                 │ 13 ball │
                 │  joints  │   ← 8 tension-only tendons
                 │          │
                 ╰────┬────╯
                    tip site (z≈0.60)  → draws in XY
```

## Model

Generate the tip-up model (needs the Onshape credentials from `.env`):

```bash
python -m mujoco_to_mjcf.build --out urdf/gradus/tracer --tip-up
```

This is the normal 14-station model with:

- a **180° flip about world X** (foot points up, hip stays at the origin),
- a dedicated `tip` site at the foot centre,
- the usual 8 servos on tendon length, 13 ball joints and rod stiffness.

`urdf/gradus/tracer/` is gitignored.

## Train

```bash
python -m mujoco_leg.train --minutes 55
```

Defaults: 8192 parallel MJX envs, 16 control steps per update, 5 physics
substeps (100 Hz control), 4 PPO epochs, 8 minibatches, `lr=3e-4`.
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` so it uses most of the GPU.

The run directory is `logs/mujoco_leg/<run_name>/` with `metrics.csv`,
`config.json`, `best.pkl`, `latest.pkl` and `final.pkl`.

## Evaluate

```bash
python -m mujoco_leg.evaluate \
  --ckpt logs/mujoco_leg/<run>/best.pkl \
  --out logs/mujoco_leg/<run>/eval --gif
```

Held-out shapes: circle, square, star, figure-eight, plus random Fourier /
polygon curves. It writes one XY path plot per shape, a combined GIF
(perspective + top-down), and `metrics.txt`.

Example from the 20-minute, 8192-env run (19.5M transitions; checkpoint
`logs/mujoco_leg/tracer_final/final.pkl`):

| shape | RMS | max |
|---|---:|---:|
| circle | 0.95 mm | 1.52 mm |
| square | 1.31 mm | 2.66 mm |
| figure-eight | 1.19 mm | 2.07 mm |
| random Fourier/polygon | 1.0–1.5 mm | 1.8–2.3 mm |
| star | 2.77 mm | 9.41 mm |

Sharp polygon corners and the abrupt turnaround of open Bezier strokes are the
hard cases; one open random stroke peaks at ~14 mm during the reversal while
still holding ~1.5 mm RMS. Closed curves track at ~1 mm.

Evaluation artifacts are in `logs/mujoco_leg/tracer_final/eval_final/`
(`eval_circle.gif`, `eval_*.png`, `metrics.txt`).

## Watch it draw

Live viewer (top-down, 50 mm star; the training radius is 35 mm):

```bash
DISPLAY=:0 python -m mujoco_leg.view --shape star --radius 0.05 --view top
```

The policy runs on the CPU MuJoCo model with a NumPy forward pass, so the
viewer does not need JAX/CUDA. Use `--view perspective` for a 3/4 view.
On this machine GLFW needs the active X display explicitly:

```bash
DISPLAY=:0.0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    python -m mujoco_leg.view --shape star --radius 0.05
```

Render a GIF instead:

```bash
python -m mujoco_leg.render_shape \
    --shape star --radius 0.05 --view top \
    --out logs/mujoco_leg/tracer_final/draw_star.gif
```

## Task design

**Shapes** (`shapes.py`): 50 % random 2-D Fourier curves, 30 % six-vertex
polygons, 20 % open Bezier arcs. Open arcs ping-pong so the target never
teleports. Every curve is centred and scaled into a 35 mm disk.

**Observation** (131 dims): 52 joint `qpos`, 39 `qvel`, 8 relative tendon
lengths, 8 tendon tensions, tip XY, tip velocity XY, a 9-point 0.2 s
look-ahead window, and `(sin φ, cos φ)`.

**Action** (8): tendon pull fractions in `[0,1]`, mapped to servo length
targets `ctrl = L0 − stroke·pull` and low-pass filtered. Tension-only is
enforced by the servo `forcerange`.

**Reward** (XY only):

```
exp(-0.5 (‖tip_xy − target_xy‖/8 mm)²)
  − 0.5 ‖tip_xy − target_xy‖ / 50 mm
  − velocity mismatch − action smoothness − tendon tension
```

No Z term; Z is only stopped from leaving a safe band.

**PPO** (`ppo.py`): pure JAX, `jax.example_libraries.stax` MLPs (256×256),
manual Adam, GAE(γ=0.99, λ=0.95), clipped surrogate, KL early-stop.

**Physics** (`env.py`): MJX with all collisions disabled. The base is fixed
and there is no floor contact in this task, and mesh collision is far too
expensive for MJX on an 8 GB card.

## Files

| File | Role |
|---|---|
| `config.py` | env / PPO hyperparameters |
| `shapes.py` | random curve samplers, look-ahead, named eval shapes |
| `env.py` | batched MJX env: reset, observation, reward, auto-reset |
| `ppo.py` | actor-critic, GAE, Adam, PPO update |
| `train.py` | rollout + PPO loop, checkpoints, CSV metrics |
| `evaluate.py` | held-out rollouts, XY plots, GIF |
| `view.py` | live MuJoCo viewer (CPU policy, target + trail) |
| `render_shape.py` | offline GIF of a shape with target + trail |

## Notes

- The policy is **underactuated** (8 tension-only tendons, 39 DoF), so tracking
  is approximate in principle; in this workspace it converges to ~1–2 mm RMS.
- Evaluations use the deterministic mean action.
- The GIF's left pane is a perspective view; the right pane is a top-down
  close-up of the foot.
