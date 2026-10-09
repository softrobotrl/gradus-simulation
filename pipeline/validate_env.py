"""Stage 2 acceptance check: proves the generated model is a working tendon-driven sim in MJX.

For each tendon: wind its servo, simulate in MJX, and require the tip to bend toward that tendon's side.
Also checks MJX against C MuJoCo, release behaviour, and batched throughput.

Usage: python -m pipeline.validate_env pipeline/configs/opencr_tdcr_spatial.yaml
"""

import argparse
import sys
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from pipeline.env import JAXTendonLegEnv

PULL_ANGLE = 0.3  # rad of drum rotation on the pulled tendon; the others pay out half that
SETTLE_S = 0.5
COMPARE_S = 0.1  # MJX (float32) vs C (float64): short horizon, so a falling chain does not amplify the gap
COMPARE_TOL = 0.005  # fraction of chain height; joint frictionloss (stick-slip) dominates the gap


def make_simulate_mjx(env: JAXTendonLegEnv, seconds: float):
    m = env.mj_model
    n = round(seconds / m.opt.timestep)
    data0 = mjx.put_data(m, mujoco.MjData(m))

    @jax.jit
    def simulate(ctrl: jax.Array) -> mjx.Data:
        return jax.lax.fori_loop(0, n, lambda _, x: mjx.step(env.mx, x), data0.replace(ctrl=ctrl))

    return lambda ctrl: simulate(jp.array(ctrl, dtype=jp.float32))


def simulate_c(env: JAXTendonLegEnv, ctrl: np.ndarray, seconds: float) -> mujoco.MjData:
    m = env.mj_model
    data = mujoco.MjData(m)
    data.ctrl[:] = ctrl
    mujoco.mj_step(m, data, nstep=round(seconds / m.opt.timestep))
    return data


def check(config_path: str) -> bool:
    env = JAXTendonLegEnv(config_path)
    m, r, tip = env.mj_model, env.cfg.servo.drum_radius, env.tip
    results: list[tuple[str, bool, str]] = []
    simulate_mjx = make_simulate_mjx(env, SETTLE_S)

    rest = simulate_mjx(np.zeros(m.nu))
    rest_tip = np.array(rest.site_xpos[tip])
    sag = np.linalg.norm(rest_tip[:2])
    results.append(("rest: stable under gravity", bool(np.isfinite(rest_tip).all()),
                    f"tip xy offset {sag * 1e3:.2f} mm, height {(rest_tip[2] - env.base_z) * 1e3:.1f} mm"))

    holes = np.array([m.site(f"disk0_hole{k}").pos[:2] for k in range(m.nu)])
    for k in range(m.nu):
        ctrl = np.full(m.nu, -PULL_ANGLE / 2)
        ctrl[k] = PULL_ANGLE
        d = simulate_mjx(ctrl)
        moved = np.array(d.site_xpos[tip]) - rest_tip
        toward = float(moved[:2] @ holes[k] / np.linalg.norm(holes[k]))
        tension = float(d.actuator_force[k]) / r
        ok = bool(np.isfinite(moved).all() and toward > 1e-3)
        results.append((f"pull tendon{k} ({PULL_ANGLE} rad = {PULL_ANGLE * r * 1e3:.1f} mm cable)", ok,
                        f"tip moved {toward * 1e3:.1f} mm toward its side, tension {tension:.1f} N"))
        if k == 0:
            c = simulate_c(env, ctrl, COMPARE_S)
            short = make_simulate_mjx(env, COMPARE_S)(ctrl)
            err = float(np.abs(c.site_xpos[tip] - np.array(short.site_xpos[tip])).max())
            results.append(("MJX matches C MuJoCo", err < COMPARE_TOL * env.height,
                            f"max tip error {err * 1e6:.1f} um after {COMPARE_S} s"))

    slack = simulate_mjx(np.full(m.nu, m.actuator_ctrlrange[0, 0]))
    tension = np.array(slack.actuator_force) / r
    results.append(("release: tendons go slack", bool(np.abs(tension).max() < 0.5),
                    f"max tension {np.abs(tension).max():.2f} N"))

    n_envs, n_steps = 64, 25
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    state = step(reset(keys), jp.zeros((n_envs, m.nu)))
    jax.block_until_ready(state.obs)
    start = time.perf_counter()
    for _ in range(n_steps):
        state = step(state, jax.random.uniform(keys[0], (n_envs, m.nu), minval=-1, maxval=1))
    jax.block_until_ready(state.obs)
    rate = n_envs * n_steps * env.n_substeps / (time.perf_counter() - start)
    results.append((f"batched env ({n_envs} envs, {jax.default_backend()})", bool(np.isfinite(state.obs).all()),
                    f"{rate:,.0f} physics steps/s, obs size {env.obs_size}"))

    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    sys.exit(0 if check(parser.parse_args().config) else 1)
