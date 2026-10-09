"""Minimal PyTorch training example: PPO from rsl-rl on TendonLegEnv (same setup as go2_train.py).

Logs and checkpoints go to logs/<exp-name>/; watch them with `tensorboard --logdir logs`.

Usage: python -m pipeline.examples.train_pytorch pipeline/configs/opencr_tdcr_spatial.yaml --iterations 100
"""

import argparse

from rsl_rl.runners import OnPolicyRunner

from pipeline.env import TendonLegEnv


def ppo_cfg(exp_name: str) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128], "activation": "elu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval": 50,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def main(config: str, exp_name: str, num_envs: int, iterations: int) -> None:
    env = TendonLegEnv(config, num_envs=num_envs)
    runner = OnPolicyRunner(env, ppo_cfg(exp_name), f"logs/{exp_name}", device=env.device)
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--exp-name", default="tendon-leg")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    main(args.config, args.exp_name, args.num_envs, args.iterations)
