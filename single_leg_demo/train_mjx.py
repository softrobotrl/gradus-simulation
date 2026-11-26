"""Train the single-leg policy with Brax PPO on the MJX env (fast on a GPU).

    python -m single_leg_demo.train_mjx --run-name mjx_v5 --dr --rail-friction 3 \\
        --cmd-min 0.05 --cmd-max 0.08 --cmd-period 3 --set cmd_period_jitter=0.333 \\
        --episode-steps 600 --num-envs 4096 --timesteps 200_000_000

Same environment options as train.py (reward preset, --set, --dr, ...).  The
CPU env stays the reference: the run saves `policy.npz` (plain arrays) plus
`config.json`, so `reward_check/passes.py RUN/policy.npz` judges it on the
CPU env, and evaluate.load_policy can run it without JAX.

Differences from train.py (rsl-rl): Brax PPO; episode starts come from a bank
of CPU-env resets (mjx_env.build_reset_bank) at full randomization, so there
is no --dr-ramp or command warm-up; no clone / KL anchor yet.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from single_leg_demo import rewards
from single_leg_demo.rig_env import RigConfig


def export_policy(params, path: Path, activation: str) -> None:
    """Save the actor (input normalizer + MLP) as plain numpy arrays."""
    norm, policy = params[0], params[1]
    layers = policy["params"]
    names = sorted((k for k in layers if k.startswith("hidden_")), key=lambda k: int(k.split("_")[1]))
    arrays = {"obs_mean": np.asarray(norm.mean["state"]), "obs_std": np.asarray(norm.std["state"]),
              "activation": np.array(activation)}
    for i, n in enumerate(names):
        arrays[f"w{i}"] = np.asarray(layers[n]["kernel"])
        arrays[f"b{i}"] = np.asarray(layers[n]["bias"])
    np.savez(path, **arrays)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--timesteps", type=float, default=100e6)
    p.add_argument("--episode-steps", type=int, default=600)
    p.add_argument("--unroll", type=int, default=24, help="steps per env per update (train.py: 24)")
    p.add_argument("--minibatches", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4, help="updates per batch")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--bank", type=int, default=4096, help="CPU-env episode starts to sample from")
    p.add_argument("--evals", type=int, default=20, help="progress reports over the run")
    p.add_argument("--reward", choices=sorted(rewards.PRESETS), default=rewards.DEFAULT_PRESET)
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    p.add_argument("--dr", action="store_true")
    p.add_argument("--rail-friction", type=float, default=None)
    p.add_argument("--cmd-min", type=float, default=None)
    p.add_argument("--cmd-max", type=float, default=None)
    p.add_argument("--cmd-period", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", default="", metavar="PROJECT", help="also log to Weights & Biases")
    args = p.parse_args()

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from flax import linen

    from single_leg_demo import mjx_env

    cfg = RigConfig(num_envs=1, num_threads=8, episode_steps=args.episode_steps, seed=args.seed,
                    curriculum_iter_steps=args.unroll, dr_enabled=args.dr)
    rewards.apply_preset(cfg, args.reward)
    for key in ("rail_friction", "cmd_min", "cmd_max", "cmd_period"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    rewards.apply_overrides(cfg, args.set)
    cfg.obs_cmd_scale = cfg.cmd_max

    run_dir = Path("logs/single_leg_demo") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = {k: v for k, v in vars(args).items() if k not in ("set",)}
    (run_dir / "config.json").write_text(json.dumps(
        {"environment": asdict(cfg), "training": {"trainer": "brax_ppo", **train_cfg}}, indent=2))
    print(f"* run {run_dir}; devices {jax.devices()}", flush=True)

    t0 = time.time()
    bank = mjx_env.build_reset_bank(cfg, args.bank, seed=args.seed)
    env = mjx_env.MjxRigEnv(cfg, bank=bank)
    print(f"* reset bank of {args.bank} in {time.time() - t0:.0f} s; obs {env.num_obs}, "
          f"critic {env.num_critic_obs}, actions {env.action_size}", flush=True)

    logger = None
    if args.wandb:
        import wandb
        logger = wandb.init(project=args.wandb, name=args.run_name,
                            config={"environment": asdict(cfg), "training": train_cfg})

    start = time.time()

    def progress(step, metrics):
        keep = {k: float(v) for k, v in metrics.items()
                if k.startswith(("eval/episode_reward", "eval/episode_r_", "eval/episode_failed",
                                 "eval/episode_vx", "eval/episode_v_track", "training/sps",
                                 "eval/avg_episode_length"))
                and not k.endswith("_std")}
        line = (f"[{time.time() - start:6.0f} s] steps {step:>12,}  "
                f"reward {keep.get('eval/episode_reward', float('nan')):8.2f}  "
                f"len {keep.get('eval/avg_episode_length', float('nan')):5.0f}  "
                f"sps {keep.get('training/sps', float('nan')):9.0f}")
        print(line, flush=True)
        with open(run_dir / "progress.jsonl", "a") as f:
            f.write(json.dumps({"step": int(step), "time": time.time() - start, **keep}) + "\n")
        if logger:
            logger.log(keep, step=int(step))

    activation = "elu"
    net = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256), value_hidden_layer_sizes=(256, 256),
        activation=linen.elu, policy_obs_key="state", value_obs_key="privileged_state",
        distribution_type="normal", noise_std_type="scalar", init_noise_std=args.init_std)

    def save(step, make_policy, params):
        export_policy(params, run_dir / "policy.npz", activation)

    batch = args.num_envs // args.minibatches
    make_inference, params, _ = ppo.train(
        environment=env, num_timesteps=int(args.timesteps), num_envs=args.num_envs,
        episode_length=args.episode_steps, wrap_env_fn=mjx_env.wrap_for_training(env),
        learning_rate=args.learning_rate, entropy_cost=args.entropy, discounting=0.99,
        gae_lambda=0.95, unroll_length=args.unroll, batch_size=batch,
        num_minibatches=args.minibatches, num_updates_per_batch=args.epochs,
        normalize_observations=True, clipping_epsilon=0.2, max_grad_norm=1.0,
        network_factory=net, seed=args.seed, num_evals=args.evals,
        num_eval_envs=min(128, args.num_envs), progress_fn=progress, policy_params_fn=save)
    export_policy(params, run_dir / "policy.npz", activation)
    print(f"* saved {run_dir / 'policy.npz'}", flush=True)


if __name__ == "__main__":
    main()
