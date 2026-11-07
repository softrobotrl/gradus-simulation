# The leg in SOFA: FEM fibreglass rod + rigid disks + tendon servos

`sofa_leg/` is the high-fidelity version of the MuJoCo leg. It loads the same merged `leg.xml` from the
Onshape→MJCF pipeline (Daniel's `urdf/gradus/sim/leg.xml`), with the same disks, masses, tendon routes and
servos. The one big difference is the backbone:

| | MuJoCo (`leg.xml`) | SOFA (`sofa_leg/`) |
|---|---|---|
| Backbone | 13 ball joints, each a spring k = EI / pitch = 128.2 N·m/rad (same for bend and twist) | A continuous FEM rod: 7.9375 mm fibreglass, E = 30 GPa, ν = 0.3, hexahedral elements, corotational (large rotations) |
| Disk ↔ rod | Disks *are* the links | Rigid disks clamped onto the rod; rod nodes inside a disk move with it, exact two-way coupling |
| Twist stiffness | Same as bending (one spring) | From the material: G = E / (2(1+ν)), about 23% softer than bending per unit angle for a round rod |
| Where the rod bends | At 13 points (the joints) | Everywhere between disks (25 mm of free rod between neighbours with the default grip) |
| Rod mass | None | 59 g (1900 kg/m³, assumed) on top of the disks' 1236 g |
| Joint friction | `frictionloss = 0.1` N·m per joint | None (no equivalent in a rod) |
| Damping | `damping = 0.5` N·m·s/rad per joint | Rayleigh, β = 0.5 / 128.2 = 0.0039 s, so each bending mode gets the same damping ratio as in MuJoCo |
| Tendons | Spatial tendons through sites, frictionless | SoftRobots `CableConstraint` through the same holes, frictionless |
| Servos | `<position>` on tendon length, kp 5000, kv 50, tension 0–200 N | The same law, applied as a cable tension every step |
| Step | 2 ms, Euler | 1 ms, implicit Euler |

Nothing about the leg is re-derived from the CAD: `sofa_leg/spec.py` compiles `leg.xml` with MuJoCo and reads every
disk pose, mass, inertia, mesh, hole position and servo parameter from it. Only the rod has to be described
separately, because `leg.xml` only holds its stiffness. Its diameter and E are the values Daniel's pipeline used to
compute that stiffness (`mujoco_to_mjcf/config.json`); ν and density are assumptions.

## Setup (Windows)

1. Install **SOFA v26.06.00** for Windows from https://www.sofa-framework.org/download/ (the installer includes
   SofaPython3, SoftRobots and STLIB). Below it is at `C:\Users\<you>\SOFA\v26.06.00`.
2. Make the Python env. SOFA's bindings are built for **Python 3.12 only**, so this is separate from `gradus-rl`:

   ```powershell
   conda env create -f environment-sofa.yml
   conda activate sofa
   $S = "C:\Users\<you>\SOFA\v26.06.00"
   conda env config vars set SOFA_ROOT=$S PYTHONPATH="$S\plugins\SofaPython3\lib\python3\site-packages;$S\plugins\SoftRobots\lib\python3\site-packages;$S\plugins\STLIB\lib\python3\site-packages"
   conda deactivate; conda activate sofa
   python -c "import Sofa, SofaRuntime, softrobots; print('ok')"
   ```

   You do not need SOFA's `bin` on PATH: `import Sofa` finds its DLLs from `SOFA_ROOT`. The message
   `SOFAPYTHON3_ROOT is empty. Trying to guess it` is harmless. `conda run` cannot pass multi-line `-c` code; use
   files or an activated shell.
3. `urdf/gradus/sim/leg.xml` (gitignored) must be there, as for the Genesis leg.

## Run it

All from the repo root, in the activated `sofa` env.

```powershell
python -m sofa_leg.play --seconds 2                    # headless circle drive: prints sweep, speed, tensions
python -m sofa_leg.play --gui                          # SOFA's ImGui viewer (untested by me, see below)
& "$env:SOFA_ROOT\bin\runSofa.exe" -l SofaPython3 sofa_leg\run_scene.py   # runSofa viewer
python -m sofa_leg.checks                              # rod physics vs beam theory (~1 min)
python -m sofa_leg.validate                            # SOFA vs MuJoCo, both grip modes (~30 min)
python -m sofa_leg.collect --workers 13 --paired-mujoco   # surrogate training data (see "Surrogate data")
python -m unittest tests.test_sofa_leg -v              # 15 tests
```

In the viewer the leg follows the same circle drive as the MuJoCo and Genesis players. Keys (the 3D view needs focus;
some SOFA GUIs only pass keys while Ctrl is held): **T** switches between the circle and keyboard teleop, the
**arrow keys** pull the front/back/left/right inner tendon 10% more per press, **O** adds the outer tendons at 45%,
**R** releases everything. The viewer runs far slower than real time (see Speed).

## How it is built

`sofa_leg/scene.py` builds this graph:

```
root                     gravity, FreeMotionAnimationLoop, BlockGaussSeidelConstraintSolver
└─ leg                   EulerImplicitSolver (Rayleigh β), SparseLDLSolver, GenericConstraintCorrection
   ├─ free               the rod nodes no disk holds (independent DOFs)
   ├─ disks              one Rigid3d frame per disk at its centre of mass (independent DOFs); hip mount fixed
   │  ├─ body_<i>        that disk's mass and inertia (SOFA allows one mass per node, hence one node each)
   │  ├─ clamped         rod nodes inside a disk, RigidMapping: they move with their disk
   │  ├─ holes           every tendon hole, RigidMapping; one CableConstraint per tendon
   │  └─ visual_<i>      the disk's STL (viewer only)
   └─ rod                all rod nodes, SubsetMultiMapping(free + clamped):
                         HexahedronFEMForceField (corotational) + MeshMatrixMass
```

* **Two-way coupling.** Each disk is a real rigid body. The rod's nodes inside it are not independent: they are
  mapped onto the disk, so the rod's elastic forces act on the disk and the disk's weight and tendon loads bend the
  rod. This is STLIB's "rigidification" pattern (`stlib3/physics/mixedmaterial/rigidification.py`), written out so
  each disk keeps its own mass. `python -m sofa_leg.checks` proves it: a 1 kg disk hung on a cantilever sags
  exactly as beam theory says (with one-way coupling it would fall freely).
* **Grip.** How a disk holds the rod is the biggest open modelling question, so it is a setting
  (`RodParams.grip`):
  * `thickness` (default): every rod node inside the disk's 19.6 mm thickness is clamped. Between neighbouring
    disks 25 mm of rod bends. Physical if the rod is a snug or glued fit in the bore (bore radius 4.1 mm vs rod
    3.97 mm).
  * `plane`: only the node layer at each disk's mid-plane is clamped, so the full 44.6 mm pitch bends. This is
    the assumption behind MuJoCo's k = EI / pitch. A bending span 1.8× longer is about 1.8× softer.
* **Rod mesh** (`sofa_leg/rodmesh.py`). 3 × 3 nodes across the rod, mapped from a square onto the disk, then scaled
  so the mesh's second moment of area equals the round rod's exactly (and density scaled so mass per metre is
  exact). Element length 1.5 mm where the rod bends, 5 mm inside disks. Node layers sit exactly on every disk face.
* **Tendons** are SoftRobots `CableConstraint`s in *force* mode through each tendon's holes, hip hole first, so a
  cable's length is the same sum of hole-to-hole distances as MuJoCo's spatial tendon. On SOFA v26.06 the force-mode
  `value` is an **impulse per step** (tension × dt), not a force; measured by holding a 1 kg body against gravity.
* **Servos.** `ServoController` sets each tension every step from MuJoCo's position-servo law,
  `tension = clip(kp (L − ctrl) + kv dL/dt, 0, Fmax)`, using hole positions and velocities computed from the disk
  frames. `tests/test_sofa_leg.py` checks the law against MuJoCo's `actuator_force` on random poses.

## Results

All numbers below come from `urdf/gradus/sim/leg.xml` on 2026-11-07, using SOFA v26.06.00 and mujoco 3.14.0. The
full outputs are in `pipeline/build/sofa/validate_<grip>.json`, which is gitignored; rerun
`python -m sofa_leg.validate` to regenerate them.

**Rod physics (`python -m sofa_leg.checks`).** This check bends a rod-only cantilever under gravity and compares
the sag with beam theory, qL⁴/(8EI). Linear hexahedra are slightly too stiff in bending (shear locking); the error
shrinks as the elements get smaller:

| Element length (3 × 3 section) | 5 mm | 2.5 mm | **1.5 mm (default)** | 1 mm | 1.5 mm, 4 × 4 section |
|---|---|---|---|---|---|
| FEM sag / beam theory | 0.796 | 0.907 | **0.935** | 0.944 | 0.959 |

So the default rod is about 7% too stiff where it bends.

The **disk-hang check** hangs a 1 kg rigid disk on the end of a 0.2 m rod. It sags 4.821 mm, against 4.824 mm from
beam theory on the same mesh. The disk's weight bends the rod and the rod holds the disk up, so the coupling works
both ways. The check's exit code fails if the error is over 5%.

**SOFA vs MuJoCo (`python -m sofa_leg.validate`).** The gated checks all pass for both grips:
- Total disk mass is 1236.2 g in both models.
- Rest tendon lengths match to 1e-13 mm.
- Every value is finite.
- Every pull moves the foot in MuJoCo's direction (cosine ≥ 0.9997) and by 0.5–2× as far.

| | MuJoCo | SOFA, grip = `thickness` (default) | SOFA, grip = `plane` |
|---|---|---|---|
| Foot motion, one inner tendon 25 / 50 / 100% | 13.3 / 26.6 / 53.5 mm | 7.9 / 15.8 / 31.6 mm (0.59×) | 12.0 / 24.1 / 48.3 mm (0.90×) |
| Servo tension at 100% | 141.9 N | 148.2 N | 142.8 N |
| Droop with gravity turned sideways | 42.3 mm | 24.3 mm | 38.6 mm |
| Circle drive (0.5 amplitude, 0.35 Hz): foot path size | 115.0 mm | 72.8 mm | 111.4 mm |
| Circle drive: RMS foot distance from MuJoCo | | 15.0 mm | 5.1 mm |
| Rod nodes (free) | | 2556 (1872) | 3780 (3600) |

Notes:

* All four tendon directions give the same numbers because the leg is symmetric, so the table shows one.
* MuJoCo's joint friction (`frictionloss`) barely matters here. Without it, the 25% pull moves the foot 13.30 mm
  instead of 13.28 mm, and the droop is 42.27 mm instead of 42.34 mm.
* **With `plane`, the two models agree to about 10%.** About 7 points of that gap comes from the FEM mesh being too
  stiff (see the rod-sag table above). The rest is real physics: a continuous rod is not a chain of point springs,
  its twist is softer, and it has mass. So under MuJoCo's own assumption about where the rod bends, SOFA reproduces
  MuJoCo.
* **With `thickness`, the leg is about 1.7× stiffer.** Only 25 mm of rod bends between disks instead of the full
  44.6 mm pitch.
* **Which grip is right depends on how the real disks hold the rod, so it needs a hardware measurement.** Hang a
  known weight on the real leg's foot, or pull one tendon with a known force, and compare the deflection with the two
  grips. That one number settles the default. It also tells the MuJoCo model whether its joint stiffness (EI / pitch)
  should use the free rod length instead of the pitch.

## Surrogate data

`python -m sofa_leg.collect` writes training data for a one-step surrogate f(s, a) → s′. It follows the paper the
MBRL track is based on (Niu et al., arXiv 2406.07069, Sec. 3.3): sequences of random actions from one fixed start
state, with a few "expert" sequences mixed in.

**The dataset that has already been collected** is in `data/sofa_leg/` on Andy's machine (gitignored). It came from
this command:

```powershell
python -m sofa_leg.collect --workers 13 --paired-mujoco     # all defaults below; 99.5 min on 14 cores
```

| File | Contents |
|---|---|
| `sofa_leg.npz` | `s` (250, 100, 20), `a` (250, 100, 8), `s_next` (250, 100, 20), all float64 |
| `sofa_leg_meta.json` | The team fields `sim, dog, dt, state_dim, action_dim, n_seq, seq_len, expert_frac, seed`, plus `state_names`, `action_names`, `action_to_pull`, `sim_dt`, `control_dt`, `dropped`, the rod and servo parameters, `xml_sha256`, `git_commit` and `wall_s` |
| `sofa_leg_raw.npz` | Per-control-step traces in float32: all 14 disk poses (position + quaternion xyzw) and velocities; tip and mid-disk position and velocity; tendon length, rate and tension; plus `action`, `is_expert` and `seq_index`. Use this to build a different state without rerunning SOFA |
| `sofa_leg_mujoco*.{npz,json}` | The **same action sequences** replayed in MuJoCo on `leg.xml`, in the same format. Its raw file has `qpos`/`qvel` instead of disk poses |

* **Format.** The layout matches the team's earlier collector (`surrogate/collect.py`, commit `552e808`): an
  `np.savez_compressed` of `s, a, s_next`, each shaped (sequences, steps, dim), plus `<out>_meta.json` with the same
  fields. Loaders written for `surrogate.py` should read it unchanged. `s_next[:, t] == s[:, t + 1]` holds exactly.
  * **Overlap with Daniel and Nate's work:** that collector belongs to the surrogate + policy track and is no longer
    on `main`. This code does not import it; it only matches its file layout and defaults. If their loader has
    changed since, this collector is the one to update.
* **State (20 numbers).** SI units, world frame; the leg hangs along −z from the hip. `state_names` lists them in
  order:
  * foot tip position and velocity (6)
  * position and velocity of the centre of mass of disk `prt_segment1_diskr_6`, where the outer tendons end (6)
  * the 8 tendon lengths (8)
* **Actions (8).** One per tendon servo, in `action_names` order: inner front/left/back/right, then the outer four.
  * Each action is drawn uniformly from [−1, 1], fresh every control step.
  * It maps to `pull = (a + 1) / 2`, where 0 is the servo at rest and 1 is full stroke (the same as
    `MujocoLeg.set_pull`).
  * Each action is held for `control_dt` = 0.02 s, which is 20 SOFA steps of 1 ms.
* **Sequences.** There are 250 sequences of 100 steps (2 s each), all starting from s0, the leg settled at rest under
  gravity.
  * Each sequence is an expert one with probability `expert_frac` = 2% (10 of the 250 here). An expert sequence is a
    circle drive with random amplitude 0.3–0.8, frequency 0.25–1 Hz and phase.
  * Sequence i is seeded by `(seed, i)`, so any subset can be regenerated.
  * No sequences were dropped for non-finite values.
* **Coverage.** Across the dataset the foot moves through about 168 × 151 mm sideways. MuJoCo's foot covers
  248 × 256 mm on the same actions; the difference is the stiffness gap described in Results. On average the foot is
  25 mm from s0. Random actions mostly cancel each other out, so large motions come mainly from the expert sequences.
  If the surrogate does poorly away from rest, raise `--expert-frac`.
* **Options.**
  * `--n-seq`, `--seq-len`, `--control-dt`, `--expert-frac`, `--seed`, `--out`
  * `--grip plane`: MuJoCo-like stiffness, about 2× slower
  * `--randomize`: varies rod E and servo kp per sequence by ±20%, with the factors saved in the raw file, for
    robustness
  * `--save-nodes`: adds every FEM node position to the raw file, which makes it large

To load it:

```python
import json, numpy as np
d = np.load("data/sofa_leg/sofa_leg.npz")
s, a, s_next = d["s"], d["a"], d["s_next"]                  # (250, 100, 20), (250, 100, 8), (250, 100, 20)
meta = json.load(open("data/sofa_leg/sofa_leg_meta.json"))
X = np.concatenate([s, a], -1).reshape(-1, 28); Y = s_next.reshape(-1, 20)   # one-step pairs
```

## Speed

SOFA is slow because the rod's stiffness matrix is factorised again every step. On Andy's laptop (14 cores, one
process per leg):

| | Steps / s at dt = 1 ms | Speed vs real time |
|---|---|---|
| grip = `thickness` (2556 nodes) | about 10.6 | 0.011× |
| grip = `plane` (3780 nodes) | about 5.9 | 0.006× |

Running several processes in parallel scales with the number of cores, since each process has its own scene. The
250 × 100 dataset (500 s of leg time) took 99.5 min on 13 processes. `python -m sofa_leg.validate` takes about
10 min for `thickness` plus 17 min for `plane`.

dt = 1 ms was chosen by a convergence test: 0.4 s of circle drive compared against a 0.5 ms reference.
- 1 ms drifts 0.37 mm RMS, which is 0.9% of the motion.
- 2 ms drifts 1.06 mm, which is 2.5%.

Settling to rest uses a 4 ms step, since only the end state matters there.

## Known limits

* **The disk grip is unknown** (see Results). It changes the leg's stiffness by about 1.7×, more than anything else
  here.
* **The FEM rod is about 7% too stiff** in bending at the default mesh, from shear locking in linear hexahedra.
  Smaller elements help only slowly and cost a lot. If needed, use `RodParams(free_element=..., n_side=4)`.
* **ν = 0.3 and density 1900 kg/m³ are assumed** (typical fibreglass); only d and E come from the pipeline. Twist
  stiffness depends on ν, whereas MuJoCo uses one spring for both bend and twist.
* **Rod mass.** SOFA's rod weighs 59 g, but MuJoCo's leg has no rod mass. That is small next to the 1236 g of disks.
* **No friction.** There is no tendon friction in the holes (neither model has it). There is no joint friction
  either, because a rod has no equivalent of `frictionloss`; in any case, MuJoCo's friction barely changes its
  results.
* **The viewer hasn't been checked by eye.**
  * `runSofa ... run_scene.py` loads and steps: 30 steps in batch mode, with no errors.
  * `play.py --gui` is written, but nobody has opened the window or tried the keys yet.
  * Arrow key codes differ between SOFA GUIs. If the arrows do nothing, use the circle drive, or report the printed
    key.
* **One leg per process, with no batching.** That is fine for data collection with `--workers`, but not for RL with
  SOFA in the loop.
* **Restoring a state** (`SofaLeg.set_state`) is repeatable to about 1e-6 m only with `flush=True`, the default.
  SOFA keeps solver state outside the MechanicalObjects, so the restore steps once and then writes the state again.

## Files

| File | |
|---|---|
| `spec.py` | Reads `leg.xml` through MuJoCo: disks, holes, servos, tip; `RodParams` for the rod (no SOFA needed) |
| `rodmesh.py` | Hexahedral mesh of a round rod with exact second moment of area (numpy only) |
| `scene.py` | Builds the SOFA scene; `ServoController`; `servo_tension` |
| `leg.py` | `SofaLeg`: reset / set_pull / step / tip / tension / length / get_state / set_state |
| `play.py` | Headless circle drive, CSV export, `--gui` |
| `run_scene.py` | runSofa entry with circle drive and keyboard teleop |
| `checks.py` | Rod sag vs beam theory at several mesh sizes; disk-hang coupling check |
| `validate.py` | SOFA vs MuJoCo on the same `leg.xml` |
| `collect.py` | Surrogate training data in the team's `(s, a, s_next)` format, optionally paired with MuJoCo |
| `../environment-sofa.yml` | The `sofa` conda env (Python 3.12 + mujoco) |
| `../tests/test_sofa_leg.py` | Spec, mesh, servo law, scene tests |
