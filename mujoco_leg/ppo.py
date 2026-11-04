"""Minimal PPO in pure JAX (no flax/optax required)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.example_libraries import stax

from mujoco_leg.config import Config


def init_params(key: jax.Array, obs_dim: int, act_dim: int, hidden: int):
    k_pi, k_vf = jax.random.split(key)
    pi_init, pi_apply = stax.serial(
        stax.Dense(hidden), stax.Tanh,
        stax.Dense(hidden), stax.Tanh,
        stax.Dense(2 * act_dim),
    )
    vf_init, vf_apply = stax.serial(
        stax.Dense(hidden), stax.Tanh,
        stax.Dense(hidden), stax.Tanh,
        stax.Dense(1),
    )
    _, pi_params = pi_init(k_pi, (1, obs_dim))
    _, vf_params = vf_init(k_vf, (1, obs_dim))
    params = {"pi": pi_params, "vf": vf_params}
    return params, (pi_apply, vf_apply)


def _dist(pi_apply, params, obs):
    out = pi_apply(params["pi"], obs)
    mean, log_std = jnp.split(out, 2, axis=-1)
    log_std = jnp.clip(log_std, -2.0, 0.5)
    return mean, log_std


def policy_step(params, apply, obs, key):
    pi_apply, vf_apply = apply
    mean, log_std = _dist(pi_apply, params, obs)
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    action = jnp.tanh(u)
    logp = -0.5 * jnp.sum(
        ((u - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1
    )
    logp = logp - jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
    value = vf_apply(params["vf"], obs)[..., 0]
    return action, logp, value, mean


def policy_mean(params, apply, obs):
    pi_apply, _ = apply
    mean, _ = _dist(pi_apply, params, obs)
    return jnp.tanh(mean)


def value_fn(params, apply, obs):
    _, vf_apply = apply
    return vf_apply(params["vf"], obs)[..., 0]


def compute_gae(reward, done, value, last_value, gamma, lam):
    def body(carry, x):
        gae, next_value = carry
        r, d, v = x
        delta = r + gamma * next_value * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return (gae, v), gae

    (_, _), adv = jax.lax.scan(
        body, (jnp.zeros_like(last_value), last_value), (reward, done, value), reverse=True
    )
    return adv, adv + value


def adam_init(params):
    return {
        "p": params,
        "m": jtu.tree_map(jnp.zeros_like, params),
        "v": jtu.tree_map(jnp.zeros_like, params),
        "t": jnp.array(0, dtype=jnp.int32),
    }


def adam_update(state, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    tf = t.astype(jnp.float32)
    m = jtu.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jtu.tree_map(lambda v, g: b2 * v + (1.0 - b2) * g * g, state["v"], grads)
    mh = jtu.tree_map(lambda m: m / (1.0 - b1 ** tf), m)
    vh = jtu.tree_map(lambda v: v / (1.0 - b2 ** tf), v)
    p = jtu.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        state["p"], mh, vh,
    )
    return {"p": p, "m": m, "v": v, "t": t}


def _loss(params, apply, obs, action, old_logp, adv, ret, cfg: Config):
    pi_apply, vf_apply = apply
    mean, log_std = _dist(pi_apply, params, obs)
    u = jnp.arctanh(jnp.clip(action, -0.999999, 0.999999))
    logp = -0.5 * jnp.sum(
        ((u - mean) / jnp.exp(log_std)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )
    logp = logp - jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)

    ratio = jnp.exp(logp - old_logp)
    pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv))

    v = vf_apply(params["vf"], obs)[..., 0]
    vloss = 0.5 * jnp.mean((v - ret) ** 2)
    entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))

    loss = pg + cfg.vf_coef * vloss - cfg.entropy * entropy
    metrics = {
        "pg": pg,
        "vl": vloss,
        "ent": entropy,
        "kl": jnp.mean(old_logp - logp),
        "clip": jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip).astype(jnp.float32)),
    }
    return loss, metrics


def make_update(apply, cfg: Config):
    @jax.jit
    def update_minibatch(opt_state, batch):
        obs, action, old_logp, adv, ret = batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        (_, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(
            opt_state["p"], apply, obs, action, old_logp, adv, ret, cfg
        )
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jtu.tree_leaves(grads)))
        scale = jnp.minimum(1.0, cfg.max_grad_norm / (norm + 1e-6))
        grads = jtu.tree_map(lambda g: g * scale, grads)
        opt_state = adam_update(opt_state, grads, cfg.lr)
        return opt_state, metrics

    return update_minibatch
