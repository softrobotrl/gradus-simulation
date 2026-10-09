"""Minimal training example: evolution strategies (ES) on a linear policy, pure JAX, no RL library.

Shows how to batch the env with vmap and roll out with lax.scan. Not a recipe for good policies;
swap in PPO (Brax / MuJoCo Playground / rsl-rl) for real training on GPU.

Usage: python -m pipeline.train_example pipeline/configs/opencr_tdcr_spatial.yaml --generations 50
"""

import argparse

import jax
import jax.numpy as jp

from pipeline.env import JAXTendonLegEnv


def main(config: str, generations: int, population: int, sigma: float, lr: float) -> None:
    env = JAXTendonLegEnv(config, episode_s=2.0)
    n_obs, n_act = env.obs_size, env.n_act

    def episode_return(params: jax.Array, key: jax.Array) -> jax.Array:
        def act(state, _):
            state = env.step(state, jp.tanh(params @ state.obs))
            return state, state.reward

        _, rewards = jax.lax.scan(act, env.reset(key), None, length=env.max_steps)
        return rewards.mean()

    @jax.jit
    def generation(params: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        k_noise, k_env = jax.random.split(key)
        noise = jax.random.normal(k_noise, (population // 2, *params.shape))
        noise = jp.concatenate([noise, -noise])  # antithetic pairs
        returns = jax.vmap(episode_return, in_axes=(0, None))(params + sigma * noise, k_env)
        ranks = jp.argsort(jp.argsort(returns)) / (population - 1) - 0.5
        grad = jp.tensordot(ranks, noise, axes=1) / (population * sigma)
        return params + lr * grad, returns.mean()

    params = jp.zeros((n_act, n_obs))
    key = jax.random.PRNGKey(0)
    for gen in range(generations):
        key, sub = jax.random.split(key)
        params, mean_return = generation(params, sub)
        print(f"generation {gen:3d}  mean reward {float(mean_return):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()
    main(args.config, args.generations, args.population, args.sigma, args.lr)
