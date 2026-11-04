# CAD → simulation

Active pipeline: the segmented tendon-driven leg, from Onshape to a merged
MuJoCo MJCF.

Full documentation: [LEG_MJCF_PIPELINE.md](LEG_MJCF_PIPELINE.md).

## Quick start

```bash
python -m cad.build_leg_mjcf                       # writes urdf/gradus/sim/
DISPLAY=:0 python -m cad.play_leg --xml urdf/gradus/sim/scene.xml
```

## Files

| File | Role |
|---|---|
| `build_leg_mjcf.py` | entry point: clean assembly → onshape-to-robot → merge → validate |
| `build_sim_assembly.py` | builds/reuses the clean `Leg-Sim` assembly (28 bodies, 13 ball mates) |
| `merge_mjcf.py` | tendon sites, 8 spatial tendons, servos/motors, rod spring, mass |
| `play_leg.py` | animated tendon/servo controller |
| `render_leg.py` | headless pose renders |
| `leg_config.json` | document IDs + material/servo defaults |
| `export_onshape.py` | shared HMAC-signed Onshape REST client |
| `LEG_MJCF_PIPELINE.md` | full pipeline / model / actuation docs |

The generated model in `urdf/gradus/sim/` is gitignored — regenerate it on
demand. The old Gradus-dog CAD, MuJoCo/MJX environments and SAC training stack
were removed.
