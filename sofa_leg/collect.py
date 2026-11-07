"""Collect (s_t, a_t, s_{t+1}) sequences from the SOFA leg for training a surrogate dynamics model.

Method (arXiv 2406.07069, Sec. 3.3): fixed-length sequences of random actions, every one starting from the same
state s0, with ~2% "expert" sequences mixed in. A one-step model f(s_t, a_t) -> s_{t+1} is trained on them and later
stands in for the slow simulator when training the policy.

File format: the same as the team's earlier collector (surrogate/collect.py, commit 552e808), so surrogate.py's
load_dataset() reads it unchanged:
  <out>.npz          s, a, s_next   float64, shapes (S, T, state_dim), (S, T, action_dim), (S, T, state_dim)
  <out>_meta.json    sim, dog, dt, state_dim, action_dim, n_seq, seq_len, expert_frac, seed  (+ extra fields below)
Extra, so other state definitions can be built later without re-running SOFA:
  <out>_raw.npz      per sequence and control step (T + 1 samples, s0 first), float32:
                     disk_pos (S, T+1, 14, 7) centre of mass + quaternion x y z w, disk_vel (S, T+1, 14, 6),
                     tip_pos, tip_vel (S, T+1, 3), tendon_length, tendon_rate, tendon_tension (S, T+1, 8),
                     action (S, T, 8), is_expert (S,), seq_index (S,), and rod_nodes (S, T+1, N, 3) with --save-nodes

State s (20 numbers, names in the meta's state_names):
  foot tip position (3) and velocity (3), world frame, m and m/s
  position (3) and velocity (3) of the centre of mass of the disk where the outer tendons end (prt_segment1_diskr_6)
  the 8 tendon lengths, m
Action a (8 numbers in [-1, 1]): one per tendon, in leg.xml's actuator order; pull fraction = (a + 1) / 2, where 0
is the servo at rest and 1 is a full stroke. Held for one control step (default 20 ms, the team collector's dt).

With --paired-mujoco the same action sequences are replayed in MuJoCo on the same leg.xml and written as
<out>_mujoco.npz / _meta.json / _raw.npz in the same layout, so the two simulators' data can be compared one to one.

Usage:
  python -m sofa_leg.collect --n-seq 4 --seq-len 8 --out data/sofa_leg/smoke.npz          # smoke test
  python -m sofa_leg.collect --workers 12 --paired-mujoco                                    # 250 x 100 (default)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from genesis_leg.leg_play import circle_pull
from sofa_leg.spec import GRIPS, RodParams

REPO = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO / "urdf" / "gradus" / "sim" / "leg.xml"
DEFAULT_OUT = REPO / "data" / "sofa_leg" / "sofa_leg.npz"
SETTLE_S = 3.0                      # rest settle for s0 (the hanging leg is still well before this)
EXPERT_AMPLITUDE = (0.3, 0.8)       # circle-drive pull amplitude range for expert sequences
EXPERT_HZ = (0.25, 1.0)             # and frequency range, Hz
RANDOMIZE = {"youngs_modulus": (0.8, 1.2), "kp": (0.8, 1.2)}   # multiplicative ranges for --randomize


def state_names(tendon_names):
    names = [f"tip_{k}" for k in ("x", "y", "z")] + [f"tip_v{k}" for k in ("x", "y", "z")]
    names += [f"mid_{k}" for k in ("x", "y", "z")] + [f"mid_v{k}" for k in ("x", "y", "z")]
    return names + [f"len_{n.removeprefix('tendon_')}" for n in tendon_names]


def sequence_plan(seed, index, seq_len, n_act, expert_frac, control_dt, tendon_names):
    """The action sequence for sequence `index`, reproducible from (seed, index) alone."""
    rng = np.random.default_rng([seed, index])
    expert = bool(rng.random() < expert_frac)
    if not expert:
        return rng.uniform(-1.0, 1.0, size=(seq_len, n_act)), False
    amplitude, hz, phase = rng.uniform(*EXPERT_AMPLITUDE), rng.uniform(*EXPERT_HZ), rng.uniform(0, 1 / EXPERT_HZ[0])
    pulls = [circle_pull(tendon_names, phase + k * control_dt, amplitude, hz) for k in range(seq_len)]
    return 2 * np.array(pulls) - 1, True


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


# ---- SOFA side -------------------------------------------------------------------------------------------------------
_WORKER = {}


def _sofa_init(xml, rod, dt, s0):
    from sofa_leg.leg import SofaLeg

    leg = SofaLeg(xml, rod=RodParams(**rod), dt=dt)
    leg.reset()
    _WORKER.update(leg=leg, s0=s0, base_e=np.array(leg.handle.rod.fem.youngModulus.value, dtype=float, copy=True))


def _sofa_snapshot(leg):
    return {
        "disk_pos": leg.disk_frames(), "disk_vel": leg.disk_velocities(), "tip_pos": leg.tip(),
        "tip_vel": leg.tip_velocity(), "tendon_length": leg.length(), "tendon_rate": leg.length_rate(),
        "tendon_tension": leg.tension(),
    }


def _sofa_sequence(job):
    leg, s0 = _WORKER["leg"], _WORKER["s0"]
    spec = leg.spec
    mid = spec.tendons[-1].disks[-1]
    factors = job.get("factors") or {}
    fem = leg.handle.rod.fem
    if factors or job.get("was_randomized"):
        e = _WORKER["base_e"] * factors.get("youngs_modulus", 1.0)
        fem.youngModulus.value = e.tolist() if e.ndim else float(e)
        fem.reinit()
        for tendon, kp in zip(spec.tendons, job["kp_base"]):
            tendon.kp = kp * factors.get("kp", 1.0)
    leg.set_state(s0)
    steps = round(job["control_dt"] / leg.dt)
    snaps, nodes = [_sofa_snapshot(leg)], [leg.rod_nodes()] if job["save_nodes"] else []
    start = time.perf_counter()
    for a in job["actions"]:
        leg.set_pull((np.asarray(a) + 1) / 2)
        leg.step(steps)
        if not leg.finite():
            return {"index": job["index"], "ok": False, "wall": time.perf_counter() - start}
        snaps.append(_sofa_snapshot(leg))
        if job["save_nodes"]:
            nodes.append(leg.rod_nodes())
    raw = {k: np.array([s[k] for s in snaps]) for k in snaps[0]}
    raw["mid_pos"] = raw["disk_pos"][:, mid, :3]
    raw["mid_vel"] = raw["disk_vel"][:, mid, :3]
    if job["save_nodes"]:
        raw["rod_nodes"] = np.array(nodes, dtype=np.float32)
    return {"index": job["index"], "ok": True, "raw": raw, "wall": time.perf_counter() - start}


def settle_s0(xml, rod, dt, coarse_dt=0.004, finish_s=0.2):
    """The settled rest pose: SETTLE_S at a coarse step (the equilibrium does not depend on dt), then finish_s at the
    collection dt so s0 is at rest for exactly the integrator the sequences use."""
    from sofa_leg.leg import SofaLeg

    coarse = SofaLeg(xml, rod=RodParams(**rod), dt=coarse_dt)
    coarse.reset()
    coarse.step(round(SETTLE_S / coarse_dt))
    leg = SofaLeg(xml, rod=RodParams(**rod), dt=dt)
    leg.reset()
    leg.set_state(coarse.get_state(), flush=False)
    leg.step(round(finish_s / dt))
    leg.time = 0.0
    s0 = leg.get_state()
    speed = float(np.linalg.norm(leg.tip_velocity()))
    return s0, leg, speed


# ---- MuJoCo side -------------------------------------------------------------------------------------------------------
def _mujoco_collect(xml, plans, control_dt, mid_name):
    import mujoco

    from genesis_leg.leg_model import find_tip, read_routes
    from genesis_leg.leg_validate import MujocoLeg

    mj = MujocoLeg(xml)
    model, data = mj.model, mj.data
    tip_body, tip_local = find_tip(model, read_routes(model))
    tip_id, mid_id = model.body(tip_body).id, model.body(mid_name).id
    dt = float(model.opt.timestep)
    steps = round(control_dt / dt)
    if abs(steps * dt - control_dt) > 1e-9:
        raise ValueError(f"control dt {control_dt} is not a whole number of MuJoCo steps ({dt})")
    mj.reset()
    mj.step(round(SETTLE_S / dt))
    qpos0, qvel0, act0 = data.qpos.copy(), data.qvel.copy(), data.act.copy()
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def point_velocity(point, body):
        mujoco.mj_jac(model, data, jacp, jacr, point, body)
        return jacp @ data.qvel

    def snapshot():
        tip = data.xpos[tip_id] + data.xmat[tip_id].reshape(3, 3) @ np.asarray(tip_local)
        mid = data.xipos[mid_id].copy()
        return {"tip_pos": tip, "tip_vel": point_velocity(tip, tip_id), "mid_pos": mid,
                "mid_vel": point_velocity(mid, mid_id), "tendon_length": data.ten_length.copy(),
                "tendon_rate": data.ten_velocity.copy(), "tendon_tension": mj.tension(),
                "qpos": data.qpos.copy(), "qvel": data.qvel.copy()}

    out = []
    for actions in plans:
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.act[:] = qpos0, qvel0, act0
        data.ctrl[:] = model.actuator_ctrlrange[:, 1]
        mujoco.mj_forward(model, data)
        snaps = [snapshot()]
        for a in actions:
            mj.set_pull((np.asarray(a) + 1) / 2)
            mj.step(steps)
            snaps.append(snapshot())
        out.append({k: np.array([s[k] for s in snaps]) for k in snaps[0]})
    return out


# ---- assembling and saving ------------------------------------------------------------------------------------------
def compact_states(raw):
    """(T+1, 20) compact state from one sequence's raw arrays."""
    return np.concatenate([raw["tip_pos"], raw["tip_vel"], raw["mid_pos"], raw["mid_vel"], raw["tendon_length"]], axis=1)


def save(out, raws, actions, experts, indices, meta, extra_keys):
    out.parent.mkdir(parents=True, exist_ok=True)
    states = np.array([compact_states(r) for r in raws])
    acts = np.array(actions)
    np.savez_compressed(out, s=states[:, :-1], a=acts, s_next=states[:, 1:])
    raw = {k: np.array([r[k] for r in raws], dtype=np.float32) for k in extra_keys if k in raws[0]}
    raw.update(action=acts.astype(np.float32), is_expert=np.array(experts), seq_index=np.array(indices))
    np.savez_compressed(out.with_name(out.stem + "_raw.npz"), **raw)
    meta = {**meta, "n_seq": len(raws), "state_dim": states.shape[2], "action_dim": acts.shape[2]}
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    print(f"[collect] saved {states.shape[0]} x {acts.shape[1]} sequences, state_dim {states.shape[2]}, "
          f"action_dim {acts.shape[2]} -> {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-seq", type=int, default=250, help="paper optimum, Fig. 4")
    parser.add_argument("--seq-len", type=int, default=100, help="control steps per sequence")
    parser.add_argument("--control-dt", type=float, default=0.02, help="s per action (the team collector's dt)")
    parser.add_argument("--dt", type=float, default=0.001, help="SOFA physics step, s")
    parser.add_argument("--expert-frac", type=float, default=0.02, help="paper Sec. 3.3: 2%% expert sequences")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--grip", choices=GRIPS, default="thickness")
    parser.add_argument("--paired-mujoco", action="store_true", help="also replay the same actions in MuJoCo")
    parser.add_argument("--randomize", action="store_true",
                        help=f"vary rod E and servo kp per sequence by factors in {RANDOMIZE}")
    parser.add_argument("--save-nodes", action="store_true", help="also store every FEM rod node (large)")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    steps = round(args.control_dt / args.dt)
    if abs(steps * args.dt - args.control_dt) > 1e-9:
        raise SystemExit("--control-dt must be a whole number of --dt steps")

    rod = asdict(RodParams(grip=args.grip))
    t0 = time.perf_counter()
    print(f"[collect] settling s0 ({SETTLE_S:g} s at rest) ...", flush=True)
    s0, leg, speed = settle_s0(str(args.xml), rod, args.dt)
    names = leg.tendon_names
    mid_name = leg.spec.disks[leg.spec.tendons[-1].disks[-1]].name
    kp_base = [t.kp for t in leg.spec.tendons]
    print(f"[collect] s0 ready in {time.perf_counter() - t0:.0f} s, foot speed {speed:.1e} m/s", flush=True)

    plans = [sequence_plan(args.seed, i, args.seq_len, len(names), args.expert_frac, args.control_dt, names)
             for i in range(args.n_seq)]
    jobs = []
    for i, (actions, expert) in enumerate(plans):
        factors = {}
        if args.randomize:
            rng = np.random.default_rng([args.seed, i, 1])
            factors = {k: float(rng.uniform(*r)) for k, r in RANDOMIZE.items()}
        jobs.append({"index": i, "actions": actions, "control_dt": args.control_dt, "save_nodes": args.save_nodes,
                     "factors": factors, "was_randomized": args.randomize, "kp_base": kp_base})

    results, start = {}, time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_sofa_init,
                             initargs=(str(args.xml), rod, args.dt, s0)) as pool:
        futures = [pool.submit(_sofa_sequence, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results[r["index"]] = r
            if done % max(1, args.n_seq // 20) == 0 or done == args.n_seq:
                rate = done / (time.perf_counter() - start)
                print(f"[collect] {done}/{args.n_seq} sequences, {rate * 60:.1f}/min, "
                      f"~{(args.n_seq - done) / rate / 60:.0f} min left", flush=True)
    wall = time.perf_counter() - start

    kept = [i for i in range(args.n_seq) if results[i]["ok"]]
    dropped = args.n_seq - len(kept)
    commit = _git_commit()
    meta = {
        "sim": "sofa", "dog": "gradus_leg", "dt": args.control_dt, "seq_len": args.seq_len,
        "expert_frac": args.expert_frac, "seed": args.seed, "requested_n_seq": args.n_seq, "dropped": dropped,
        "state_names": state_names(names), "action_names": names,
        "action_to_pull": "pull = (a + 1) / 2, 0 = servo at rest, 1 = full stroke", "sim_dt": args.dt,
        "control_dt": args.control_dt, "s0": f"settled {SETTLE_S:g} s at rest from the straight pose",
        "rod": rod, "randomize": RANDOMIZE if args.randomize else None,
        "servo": [{"name": t.name, "kp": t.kp, "kv": t.kv, "fmax": t.fmax, "ctrl_low": t.ctrl_low,
                   "ctrl_high": t.ctrl_high} for t in leg.spec.tendons],
        "xml": str(args.xml), "xml_sha256": leg.spec.xml_sha256, "git_commit": commit,
        "sofa_root": os.environ.get("SOFA_ROOT", ""), "wall_s": wall, "workers": args.workers,
    }
    raws = [results[i]["raw"] for i in kept]
    keys = ["disk_pos", "disk_vel", "tip_pos", "tip_vel", "mid_pos", "mid_vel", "tendon_length", "tendon_rate",
            "tendon_tension"] + (["rod_nodes"] if args.save_nodes else [])
    if args.randomize:
        for i, r in zip(kept, raws):
            r["factors"] = np.array([jobs[i]["factors"][k] for k in RANDOMIZE])
        keys.append("factors")
    save(out, raws, [plans[i][0] for i in kept], [plans[i][1] for i in kept], kept, meta, keys)
    print(f"[collect] SOFA: {len(kept)} kept, {dropped} dropped (non-finite), {wall / 60:.1f} min on "
          f"{args.workers} processes", flush=True)

    if args.paired_mujoco:
        start = time.perf_counter()
        mraws = _mujoco_collect(str(args.xml), [plans[i][0] for i in kept], args.control_dt, mid_name)
        mout = out.with_name(out.stem + "_mujoco.npz")
        mmeta = {**meta, "sim": "mujoco", "sim_dt": None, "rod": None, "randomize": None, "dropped": 0,
                 "s0": f"settled {SETTLE_S:g} s at rest from qpos0", "wall_s": time.perf_counter() - start,
                 "paired_with": out.name}
        save(mout, mraws, [plans[i][0] for i in kept], [plans[i][1] for i in kept], kept, mmeta,
             ["tip_pos", "tip_vel", "mid_pos", "mid_vel", "tendon_length", "tendon_rate", "tendon_tension", "qpos",
              "qvel"])


if __name__ == "__main__":
    main()
