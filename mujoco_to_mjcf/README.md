# CAD → simulation

Active pipeline: the segmented tendon-driven leg, from Onshape to a merged
MuJoCo MJCF.

Full documentation: [PIPELINE.md](PIPELINE.md).

## Quick start

```bash
python -m mujoco_to_mjcf.build                       # writes urdf/gradus/sim/
python -m mujoco_to_mjcf.build --out urdf/gradus/tracer --tip-up   # tracer model
DISPLAY=:0 python -m mujoco_to_mjcf.play --xml urdf/gradus/sim/scene.xml
```

The play script accepts runtime servo overrides for a larger circle / more tip
lift. `--outer-scale 1.0` uses the full outer-tendon stroke, and `--servo-kp`
overrides the position gain (the current walk build uses `kp=8000`):

```bash
MUJOCO_GL=egl python -m mujoco_to_mjcf.play \
    --xml urdf/gradus/walk/scene.xml \
    --servo-kp 8000 --servo-kv 80 --outer-scale 1.0 \
    --record urdf/gradus/walk/leg_circle_lift.gif --seconds 6
```

`--tip-up` flips the model 180° about world X and adds a `tip` site at the
foot centre. The resulting `urdf/gradus/tracer/` model is used by
[`mujoco_leg/`](../mujoco_leg/README.md) for PPO shape tracing.

## Files

| File | Role |
|---|---|
| `build.py` | entry point: clean assembly → onshape-to-robot → merge → validate |
| `assembly.py` | builds/reuses the clean `Leg-Sim` assembly (28 bodies, 13 ball mates) |
| `merge.py` | tendon sites, 8 spatial tendons, servos/motors, rod spring, mass |
| `play.py` | animated tendon/servo controller |
| `render.py` | headless pose renders |
| `config.json` | document IDs + material/servo defaults |
| `onshape.py` | shared HMAC-signed Onshape REST client |
| `PIPELINE.md` | full pipeline / model / actuation docs |

The generated model in `urdf/gradus/sim/` is gitignored — regenerate it on
demand. The old Gradus-dog CAD, MuJoCo/MJX environments and SAC training stack
were removed.
