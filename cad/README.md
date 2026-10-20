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
- Overall bounding box: 510 x 294 x 310 mm including snout, tail, feet, drums and tendons.
- Four planar two-link legs in X-Z planes at `y = +/-105` mm.
- Hips at `x = +/-100`, `y = +/-105`, `z = 175` mm; feet at `z = 15` mm with 15 mm radius.
- Thigh length ~72 mm; calf length ~104 mm.
- Front legs knee forward, rear legs knee backward.
- Hip joints are passive pivots; no hip servos.
- Two antagonistic tendons per leg, eight total.
- Two tendon drums per leg, eight total, one per tendon.
- Tendon path: drum -> hip guide -> knee guide -> foot anchor.
- 25 solid bodies:
  - 1 merged body/shell (body, head, snout, ears, tail, top plate, hip forks, drum plates, guides)
  - 4 thighs
  - 4 calves
  - 8 drums
  - 8 tendons

The geometry is a concept/model, not manufacturing-ready. It intentionally omits bearing bores, fillets,
fastener details, tendon termination details, and a real servo package. The external hardware team can
replace or refine those later.

## Tendon routing

Each tendon runs over three guides, all offset in x from the leg plane. The offsets differ per tendon,
because a tendon that passes through both joints produces the same joint-torque direction as its
antagonist: the leg then has a rank-1 tendon Jacobian and cannot hold itself up.

| Tendon | Hip guide (= drum) | Knee guide | Foot anchor | Effect |
|---|---|---|---|---|
| 0 (red) | +30 mm | 0 mm (on the knee) | +30 mm | hip moment only |
| 1 (blue) | -30 mm | +30 mm | -30 mm | knee moment, opposite hip moment |

Front legs take the offsets in +x, rear legs mirror them in x. The offsets are the same ones the
simulation uses (`TENDON_GUIDES` in `pipeline/dog_model.py`), so the CAD and the MuJoCo model describe the
same mechanism. With these offsets the 2x2 tendon Jacobian of every leg has determinant 6.6e-4 and
condition number 1.9, and the dog stands in simulation.

The foot anchor sits 30 mm beside the foot, carried by a short arm (`anchorArm`); the knee guide of
tendon 1 sits 30 mm past the knee, on a bracket (`kneeBracket`).

## Exports

Current exports are in `/home/daniel/Downloads/gradus_dog/`:

- `gradus_dog.step` - STEP AP242, 25 solids
- `gradus_dog.stl` - STL mesh
- `gradus_dog_isometric.png` - shaded isometric view
- `gradus_dog_front.png` - shaded front view

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
