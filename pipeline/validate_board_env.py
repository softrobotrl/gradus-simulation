"""Acceptance check for the board-pushing task: proves the task is physically solvable before anyone trains on it.

Checks, each printed as PASS/FAIL:
  rest           the straight leg holds the board just off the rails, and the board stays put
  drop           bending the leg lowers the board onto the rails and the pad loses contact
  scripted gait  a hand-written stepping gait pushes the board forward; reversed it pushes backward;
                 rotated 90 degrees it pushes sideways (so random_direction=True is learnable too)
  random policy  random actions give near-zero net travel (the reward can't be farmed without pushing)
  MJX vs C       MJX (the GPU-capable simulator) and regular MuJoCo agree on the gait's result
  CPU backend    BoardPushEnvCPU (C MuJoCo threads) gives the same gait travel as MJX
  batched envs   both PyTorch envs report time_outs and final_observation correctly; speed per batch size

All the single-env checks run as one batch of envs in one compiled rollout, because compiling is the slow part
on a CPU (about a minute per compile).

Usage:
  python -m pipeline.validate_board_env pipeline/configs/opencr_tdcr_spatial.yaml
  python -m pipeline.validate_board_env pipeline/configs/opencr_tdcr_spatial.yaml --device gpu --sizes 256 1024 2048
"""

import argparse
import sys
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from pipeline.board_env import BoardPushEnv, BoardPushEnvCPU, JAXBoardPushEnv

GAIT_HZ = 1.0         # stepping cycles per second
GAIT_AMPLITUDE = 0.2  # rad of drum angle; sets how far the tip swings (~2 cm)
ROLLOUT_S = 4.0
MIN_TRAVEL = 0.02     # m the scripted gait must move the board in ROLLOUT_S
N_RANDOM = 16         # envs running the random policy
CONSTANT, GAIT, RANDOM = 0, 1, 2  # policy kinds


def gait_angles(env: JAXBoardPushEnv, t: jax.Array, heading: jax.Array) -> jax.Array:
    """Servo angles (rad) that sweep the tip along a circle touching the leg's axis.

    The tip passes straight up (highest, so it lifts the board) while moving along `heading`, and comes back on
    the far side of the circle bent over (lower, so the board sits on the rails while the tip returns).
    Pulling tendon k bends the tip toward its hole, so each servo gets the bend vector projected on its hole.
    """
    m = env.mj_model
    holes = np.array([m.site(f"disk0_hole{k}").pos[:2] for k in range(m.nu)])
    unit = jp.array(holes / np.linalg.norm(holes, axis=1, keepdims=True))
    phase = 2 * jp.pi * GAIT_HZ * t
    forward = jp.array([jp.cos(heading), jp.sin(heading)])
    side = jp.array([-jp.sin(heading), jp.cos(heading)])
    bend = GAIT_AMPLITUDE * (jp.sin(phase) * forward + (1 - jp.cos(phase)) * side)
    return unit @ bend


def angles_to_action(env: JAXBoardPushEnv, angles: jax.Array) -> jax.Array:
    """Inverse of the env's action mapping: +1 = ctrl_high, -1 = ctrl_low, 0 = angle 0."""
    return jp.where(angles > 0, angles / env.ctrl_high, -angles / env.ctrl_low)


def rollout(env: JAXBoardPushEnv, kinds, constants, headings, keys):
    """Run one scripted policy per env from reset for ROLLOUT_S. Returns (final states, board qpos per step).

    kinds[i] picks env i's policy: CONSTANT holds constants[i], GAIT runs gait_angles at headings[i],
    RANDOM draws uniform actions in [-1, 1] every step.
    """
    n = round(ROLLOUT_S / env.dt)

    def one(kind, constant, heading, key):
        def body(state, i):
            gait = angles_to_action(env, gait_angles(env, i * env.dt, heading))
            noise = jax.random.uniform(jax.random.fold_in(key, i), (env.n_act,), minval=-1, maxval=1)
            action = jp.select([kind == CONSTANT, kind == GAIT], [constant, gait], noise)
            state = env.step(state, action)
            return state, state.data.qpos[env.board_qpos]

        start = env.reset(key)
        state, boards = jax.lax.scan(body, start, jp.arange(n))
        return state, jp.concatenate([start.data.qpos[env.board_qpos][None], boards])

    return jax.jit(jax.vmap(one))(jp.array(kinds), jp.array(constants), jp.array(headings), keys)


def c_mujoco_gait(env: JAXBoardPushEnv, qpos0: np.ndarray, seconds: float) -> np.ndarray:
    """The forward gait in regular (float64, CPU) MuJoCo, with the env's servo rate limit. Returns board qpos."""
    m, d = env.mj_model, mujoco.MjData(env.mj_model)
    d.qpos[:] = qpos0
    target = np.full(m.nu, env.init_servo_angle)
    for i in range(round(seconds / env.dt)):
        action = angles_to_action(env, gait_angles(env, jp.array(i * env.dt), jp.array(0.0)))
        command = np.where(action > 0, action * env.ctrl_high, -action * env.ctrl_low)
        target = target + np.clip(command - target, -env.max_step, env.max_step)
        d.ctrl[:] = target
        mujoco.mj_step(m, d, nstep=env.n_substeps)
    return d.qpos[np.asarray(env.board_qpos)]


def check(config_path: str, device: str, sizes: list[int]) -> bool:
    results: list[tuple[str, bool, str]] = []
    env = JAXBoardPushEnv(config_path, init_bend_deg=0.0)  # straight start, so the scripted checks are repeatable
    gap_idx = env.obs_layout["pad_gap"]
    print(f"scene for the viewer: {env.write_scene_xml()}")

    # One batch: 2 rest envs, 2 drop envs, 3 gait headings, N_RANDOM random envs.
    zero = np.zeros(env.n_act)
    bent = np.asarray(angles_to_action(env, jp.array([0.3, -0.15, -0.15])))  # wind tendon 0, release the others
    headings = {"forward (+x)": 0.0, "reversed (-x)": np.pi, "rotated (+y)": np.pi / 2}
    kinds = [CONSTANT] * 4 + [GAIT] * 3 + [RANDOM] * N_RANDOM
    constants = [zero, zero, bent, bent] + [zero] * (3 + N_RANDOM)
    heads = [0.0] * 4 + list(headings.values()) + [0.0] * N_RANDOM
    keys = jax.random.split(jax.random.PRNGKey(0), len(kinds))
    t0 = time.perf_counter()
    final, boards = rollout(env, kinds, constants, heads, keys)
    boards = np.asarray(boards)  # (env, step, xyz)
    gaps = np.asarray(final.obs[:, gap_idx.start])
    print(f"(scripted rollouts: {len(kinds)} envs x {ROLLOUT_S:.0f} s, {time.perf_counter() - t0:.0f} s incl. compile)")

    # 1. rest: servos at 0 (tendons just taut) keep the leg straight; it should hold the board up and still.
    drift = np.abs(boards[0:2, -1, :2] - boards[0:2, 0, :2]).max()
    lift = boards[0:2, -1, 2]
    results.append(("rest: straight leg holds the board",
                    bool(np.isfinite(boards[0:2]).all() and drift < 1e-3 and (lift > 1e-3).all() and (gaps[0:2] <= 0).all()),
                    f"lift {lift.mean() * 1e3:.2f} mm, xy drift {drift * 1e3:.2f} mm, pad gap {gaps[0:2].max() * 1e3:.2f} mm"))

    # 2. drop: one tendon wound bends the leg; the tip lowers and the board should sit on the rails.
    lift = boards[2:4, -1, 2].max()
    results.append(("drop: bent leg puts the board on the rails", bool(lift < 2e-4 and (gaps[2:4] > 0).all()),
                    f"lift {lift * 1e3:.2f} mm, pad gap {gaps[2:4].min() * 1e3:.2f} mm"))

    # 3. scripted gait in three headings.
    travel = {}
    for i, (name, h) in enumerate(headings.items(), start=4):
        along = float((boards[i, -1, :2] - boards[i, 0, :2]) @ np.array([np.cos(h), np.sin(h)]))
        travel[name] = along
        results.append((f"scripted gait {name}", along > MIN_TRAVEL,
                        f"board moved {along * 1e3:.1f} mm along the heading in {ROLLOUT_S:.0f} s "
                        f"({along / ROLLOUT_S * 1e3:.1f} mm/s)"))

    # 4. random policy: net travel should average out.
    moved = boards[7:, -1, 0] - boards[7:, 0, 0]
    results.append(("random policy: no net push", abs(moved.mean()) < 0.3 * travel["forward (+x)"],
                    f"mean x travel {moved.mean() * 1e3:+.1f} mm (std {moved.std() * 1e3:.1f}) over {N_RANDOM} envs"))

    # 5. MJX vs C MuJoCo on the first second of the forward gait (float32 MJX vs float64 C reference).
    seconds = 1.0
    qpos0 = np.asarray(jax.jit(env.reset)(keys[4]).data.qpos)
    c_board = c_mujoco_gait(env, qpos0, seconds)
    mjx_board = boards[4, round(seconds / env.dt)]
    err = float(np.abs(mjx_board[:2] - c_board[:2]).max())
    moved_c = float(c_board[0] - qpos0[np.asarray(env.board_qpos)][0])
    results.append((f"MJX matches C MuJoCo ({seconds:.0f} s of gait)", err < max(2e-3, 0.25 * abs(moved_c)),
                    f"board x moved: MJX {(mjx_board[0] - boards[4, 0, 0]) * 1e3:.1f} mm, C {moved_c * 1e3:.1f} mm, "
                    f"max xy difference {err * 1e3:.2f} mm"))

    # 6. C MuJoCo backend (BoardPushEnvCPU): same task code, so the same gait must give the same travel.
    import torch

    cpu_env = BoardPushEnvCPU(config_path, num_envs=2, init_bend_deg=0.0)
    bq = np.asarray(cpu_env.jax_env.board_qpos)
    begin = np.array([d.qpos[bq] for d in cpu_env.datas])
    for i in range(round(ROLLOUT_S / env.dt)):
        action = np.asarray(angles_to_action(env, gait_angles(env, jp.array(i * env.dt), jp.array(0.0))))
        cpu_env.step(torch.from_numpy(np.tile(action, (2, 1))))
    cpu_travel = float(np.mean([d.qpos[bq][0] for d in cpu_env.datas] - begin[:, 0]))
    mjx_travel = travel["forward (+x)"]
    results.append(("C MuJoCo backend matches MJX (forward gait)", abs(cpu_travel - mjx_travel) < 0.1 * mjx_travel,
                    f"board moved {cpu_travel * 1e3:.1f} mm (MJX {mjx_travel * 1e3:.1f} mm) in {ROLLOUT_S:.0f} s"))

    # 7. both batched PyTorch envs: auto-reset bookkeeping on a short episode, then speed with random actions.
    backends = {"MJX": lambda n, **kw: BoardPushEnv(config_path, n, device=device, **kw),
                "C MuJoCo": lambda n, **kw: BoardPushEnvCPU(config_path, n, **kw)}
    for backend, make in backends.items():
        for i, n in enumerate(sizes):
            torch_env = make(n, episode_s=0.2)  # 10 control steps
            random = lambda: torch.rand(n, torch_env.num_actions, device=torch_env.device) * 2 - 1
            if i == 0:
                for _ in range(torch_env.max_episode_length):
                    obs, reward, done, extras = torch_env.step(random())
                final_obs, new_obs = extras["final_observation"], obs["policy"]
                differs = not torch.equal(final_obs, new_obs)
                ok = (bool(done.all()) and bool(extras["time_outs"].all()) and final_obs.shape == new_obs.shape
                      and bool(torch.isfinite(final_obs).all()) and differs)
                results.append((f"{backend} batched env: auto-reset bookkeeping", ok,
                                f"done {int(done.sum())}/{n}, time_outs {int(extras['time_outs'].sum())}/{n}, "
                                f"final_observation differs from the fresh obs: {differs}"))
            torch_env.step(random())  # make sure compiling is done before timing
            steps = 10
            t0 = time.perf_counter()
            for _ in range(steps):
                obs, *_ = torch_env.step(random())
            rate = n * steps / (time.perf_counter() - t0)
            jax_device = getattr(torch_env, "_jax_device", None)
            stats = (jax_device.memory_stats() if jax_device else None) or {}
            mem = f", JAX peak memory {stats['peak_bytes_in_use'] / 2**30:.2f} GB" if "peak_bytes_in_use" in stats else ""
            results.append((f"{backend} speed ({n} envs, {torch_env.device})", bool(torch.isfinite(obs["policy"]).all()),
                            f"{rate:,.0f} transitions/s with random actions{mem}"))

    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                        help="where the MJX batched env runs (auto = GPU if usable, else CPU)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64],
                        help="num_envs values to time; e.g. 256 1024 2048 on a GPU")
    args = parser.parse_args()
    sys.exit(0 if check(args.config, args.device, args.sizes) else 1)
