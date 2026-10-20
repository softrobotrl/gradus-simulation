# Gradus leg (Onshape) → Genesis

Reconstruction of the Gradus ball-socket leg for the two-tendon (computed) simulation in
[`genesis_samples/gradus_tendon.py`](../../genesis_samples/gradus_tendon.py).

## Source document

<https://cad.onshape.com/documents/571fbc8ad90a7eae9bfbfe7d/w/9f460e6d3cdf15497dabf0e2/e/4c3aaa530142f1870800c037>

| Element | ID |
|---|---|
| Leg assembly `ASY-Ball-Socket-Leg-3DP` | `d69fc4402fbb455cc35cb1ce` |
| Proximal section `ASY-Ball-Socket-3DP-S1` | `8542b08048313c4c0655dac9` |
| Distal section `ASY-Ball-Socket-3DP-S2` | `2347be89b08d6d8bf1e03f3a` |
| `PRT-Ball-Socket-Link-3DP` | `e7d5d70f2a269858e75a261b` |
| `PRT-Backbone-Link` | `95ac7c0047a0e5cbaaec91ad` |

The document is link-shared (`permission: ANONYMOUS_ACCESS`), but Onshape's geometry/export
endpoints still require an authenticated API session, so the STL/STEP meshes could not be
downloaded. The dimensions below were recovered from the public **Part Studio feature trees
and BOMs** (`/api/documents/.../elements`, `/api/partstudios/.../features`,
`/api/assemblies/.../bom`). `onshape_params.json` keeps the provenance.

## Recovered design

| Parameter | Value |
|---|---|
| Links per section | 8 (S1) + 8 (S2) = 16 |
| Joint pitch (ball-centre spacing) | **31.25 mm** |
| S1 / S2 disk diameter | **90 mm / 70 mm** |
| Disk thickness | 10 mm |
| 3D-printed ball diameter | 24 mm |
| Cable hole diameter | 4 mm |
| Cable holes per disk | 4 |
| Cable-hole pitch radius | **37.5 mm (S1)**, **29.17 mm (S2)** (= disk radius / 1.2) |
| Backbone | Ø30 mm × 500 mm |
| Nominal leg length | 16 × 31.25 = **500 mm** |

The four cable holes are what the fishing lines run through. This model uses the two holes on
the **±X** side of the disk (front/back), giving one flexor and one extensor in the sagittal
plane.

## Reconstruction (`gradus_leg.urdf`)

* `build_leg.py` generates the URDF: `python urdf/gradus/build_leg.py`.
* Each 3D ball socket is represented by a **pitch revolute joint** about `+Y`; the full 3-DOF
  ball joint is collapsed to the bending plane that the two tendons actuate. The other two
  cable holes (±Y) would be used for the lateral pair in a 3-DOF model.
* Each link is a cylinder of the correct disk radius/thickness plus a thin strut down to the
  next ball, with a lumped mass (55 g S1, 40 g S2) and cylinder inertia. A 30 g foot is fixed
  to the last disk.
* Joint limits ±1.75 rad, effort 500 N·m. Self-collision is disabled in the demo.

## Computed tendons

For a line routed through guide points `p_0 … p_N`:

```
L(q) = |a_base - p_0| + Σ |p_{i+1} - p_i| + |p_N - a_tip|
τ_i  = -(∂L/∂q_i) · F          F ≥ 0 (tension only)
```

`ComputedTendons` builds the guide points from the same analytic planar FK that the URDF uses,
then gets `∂L/∂q` with two `torch.autograd.grad` calls per step. At `q = 0` the moment arm is
exactly the cable-hole radius (37.5 mm / 29.17 mm), which matches the CAD routing.

Control is antagonistic:

```
bend = Σ q_i
diff = Kp (bend_des - bend) - Kd Σ q̇_i
F_back  = clamp(F0 + diff, 0, Fmax)     # -X line, bends q > 0
F_front = clamp(F0 - diff, 0, Fmax)     # +X line, bends q < 0
```

`F0` is co-contraction (stiffness), `diff` steers the line. The URDF has no joint springs, so
the leg is a genuine tendon-driven continuum: with no tension it hangs straight, and the two
tensions set its curvature.

## Run

```bash
conda activate gradus-rl
python genesis_samples/gradus_tendon.py                       # headless, prints tracking
python genesis_samples/gradus_tendon.py --viewer --xray       # viewer, see-through links
python genesis_samples/gradus_tendon.py --render --xray       # frames -> logs/gradus_tendon/
```

Validated on Genesis 1.2.3 (`gradus-rl`): the analytic FK matches Genesis to <0.01 mm, the
autograd tendon gradient matches finite differences to ~1e-10, and the bend tracks a ±1.1 rad
sine command (0.16 Hz) with a ~55 N co-contraction. `python genesis_samples/gradus_tendon.py`
writes `logs/gradus_tendon/results.png` (tracking + routing side view) after every run.
