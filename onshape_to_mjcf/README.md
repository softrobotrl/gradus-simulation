# CAD → simulation

Active pipeline: the segmented tendon-driven leg, from Onshape to a merged
MuJoCo MJCF.

Full documentation: [PIPELINE.md](PIPELINE.md).

## Quick start

```bash
cp .env.example .env                                  # fill in Onshape API keys
python -m onshape_to_mjcf.build                       # writes urdf/gradus/sim/
python -m onshape_to_mjcf.build --rebuild             # after CAD changes
DISPLAY=:0 python -m onshape_to_mjcf.play --xml urdf/gradus/sim/scene.xml
```

No Onshape key? Use the model zip pinned in the **#reinforcement-learning**
Discord channel.

## Files

| File | Role |
|---|---|
| `build.py` | entry point: clean assembly → onshape-to-robot → merge → validate |
| `assembly.py` | builds/reuses the clean `Leg-Sim` assembly (28 bodies, 13 ball mates) |
| `merge.py` | tendon sites, 8 spatial tendons, servos/motors, rod spring, mass |
| `play.py` | animated tendon/servo controller |
| `render.py` | headless pose renders |
| `config.json` | Onshape IDs, hole/ring radii, material/servo defaults |
| `onshape.py` | shared HMAC-signed Onshape REST client |
| `PIPELINE.md` | full pipeline / model / actuation docs |

The generated model in `urdf/gradus/sim/` is gitignored — regenerate it on
demand. The old Gradus-dog CAD, MuJoCo/MJX environments and SAC training stack
were removed.
