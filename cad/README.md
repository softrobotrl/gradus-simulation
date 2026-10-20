# Gradus Dog CAD (Onshape)

Simple tendon-driven quadruped concept, generated as a single parametric FeatureScript custom feature.

## Onshape

- Document name: `Gradus Dog`
- Document: `572c4a586468c3b5310ca798`
- Workspace: `3b9ccafe4a26826d5aea33e4`
- Part Studio: `9adb8759bcda810b416055f0`
- Feature Studio: `fc4ec8229e8ed1f926affd90`
- Part Studio: https://cad.onshape.com/documents/572c4a586468c3b5310ca798/w/3b9ccafe4a26826d5aea33e4/e/9adb8759bcda810b416055f0

The Part Studio contains one custom feature, `Gradus Dog` (`gradusDog`). The design source is
`cad/gradus_dog.fs`, which is also in the Onshape Feature Studio.

## Design

- Body shell: 300 x 160 x 80 mm, centered at `(0, 0, 215)` mm.
- Overall bounding box: 510 x 316 x 310 mm including snout, tail, feet, drums, gearboxes and tendons.
- Four planar two-link legs in X-Z planes at `y = +/-105` mm.
- Hips at `x = +/-100`, `y = +/-105`, `z = 175` mm; feet at `z = 15` mm with 15 mm radius.
- Thigh length ~72 mm; calf length ~104 mm.
- Front legs knee forward, rear legs knee backward.
- Hip joints are passive pivots; no hip servos.
- Two antagonistic tendons per leg, eight total.
- Two tendon drums per leg, eight total, one per tendon.
- Tendon path: drum -> hip guide -> knee guide -> foot anchor.
- 33 solid bodies:
  - 1 merged body/shell (body, head, snout, ears, tail, top plate, hip forks, drum plates, guides, gearboxes)
  - 4 thighs
  - 4 calves
  - 8 drums
  - 8 tendons

The geometry is a concept/model, not manufacturing-ready. It intentionally omits bearing bores, fillets,
fastener details, tendon termination details, and a real servo package. The external hardware team can
replace or refine those later.

## Tendon drive

Each tendon runs over three guides, all offset in x from the leg plane. The offsets differ per tendon,
because a tendon that passes through both joints produces the same joint-torque direction as its
antagonist: the leg then has a rank-1 tendon Jacobian and cannot hold itself up.

| Tendon | Hip guide (= drum) | Knee guide | Foot anchor | Effect |
|---|---|---|---|---|
| 0 (red) | +45 mm | 0 mm (on the knee) | +45 mm | hip moment only |
| 1 (blue) | -45 mm | +45 mm | -45 mm | knee moment, opposite hip moment |

Front legs take the offsets in +x, rear legs mirror them in x. The offsets are the same ones the
simulation uses (`TENDON_GUIDES` in `pipeline/dog_model.py`), so the CAD and the MuJoCo model describe the
same mechanism. With these offsets the 2x2 tendon Jacobian of every leg has determinant 1.3e-3 and
condition number 2.0, and the dog stands in simulation.

The offset is also the **moment arm** of the drive: the joint torque is the cable tension times the
offset, so the offsets set how much torque the drum can put into the joint. 30 mm was enough to stand
but not to walk - the knee tendon saturated at 22 N of its 35 N limit mid-stride and the leg buckled.
45 mm gives the drive about 1.5x the torque per newton of cable and lets the open-loop trot walk at
0.12 m/s without falling.

Each drum is 12 mm in radius and is driven through a 2:1 gearbox (the can behind the drum), so the
motor only has to deliver 0.25 N*m to get 0.5 N*m at the drum and 42 N of cable tension. That is the
other half of the fix: with the old 0.35 N*m direct drive the knee tendon was at 62 % of its limit while
standing and saturated as soon as the dog moved.

The foot anchor sits 45 mm beside the foot, carried by a short arm (`anchorArm`); the knee guide of
tendon 1 sits 45 mm past the knee, on a bracket (`kneeBracket`).

## Exports

Current exports are in `/home/daniel/Downloads/gradus_dog/`:

- `gradus_dog.step` - STEP AP242, 25 solids
- `gradus_dog.stl` - STL mesh (ASCII), 13,032 triangles, 510 x 294 x 310 mm
- `gradus_dog_isometric.png`, `gradus_dog_front.png`, `gradus_dog_side.png` - views of the mesh

Re-export them with `python -m cad.export_onshape` (credentials in `ONSHAPE_API`, `ONSHAPE_ACCESS_KEY`
and `ONSHAPE_SECRET_KEY`). The script retries: Onshape sometimes leaves a translation in `ACTIVE` for
minutes, answers 500 while it is busy, or fails outright during an outage (check
<https://status.onshape.com/>). On 20 October 2026 the STEP and STL exports went through during an
Onshape outage while the PNG exports kept failing; the PNGs above are therefore renders of the exported
STL made locally with `python -m cad.render_stl`, and Onshape's own shaded renders replace them when
the service is healthy again.

## Notes

- The FeatureScript uses `FeatureScript 3083` and `geometry.fs` 3083.0 to match the target Part Studio's
  library version.
- The design is intentionally simple: boxy body, two-link legs, drums, and tendon geometry. It is meant
  as a baseline for the MuJoCo/MJX model, not as final hardware CAD.
- A few `Gradus Dog API test` versions were created while debugging the Onshape API; they are snapshots
  from before the final model and can be ignored.
- The simulation side of this design lives in `pipeline/dog_model.py` (MJCF) and `pipeline/dog_env.py`
  (residual-RL environment); the SAC entry points are `policy/train_dog_sac.py` and
  `policy/play_dog_sac.py`. See `pipeline/README.md`.
