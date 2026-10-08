"""Train (headless):   python dp_train.py -n 16384
Resume latest ckpt:    python dp_train.py --resume -n 16384
Resume specific ckpt:  python dp_train.py --resume --ckpt 300
Watch latest ckpt:     python dp_train.py --eval
Specific checkpoint:   python dp_train.py --eval --ckpt 300
Smoke test:            python dp_train.py -n 64
Setup: pip install genesis-world torch tensorboard rsl-rl-lib==5.5.1
"""
import argparse
import glob
import os
import re

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from dp_env import DoublePendulumCartEnv

ENV_CFG = dict(
    urdf="robot.urdf",
    cart_joint="cart",
    joint1="shoulder_continuous",
    joint2="elbow_continuous",
    start="mixed",       # "upright" first; then "mixed" (curriculum) -> "random"
    episode_s=15.0,        # episode length in seconds; steps = episode_s / dt
)

# Config tuned for large env counts (~16k parallel environments)
TRAIN_CFG = {
    "num_steps_per_env": 128,
    "save_interval": 100,
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "algorithm": {
        "class_name": "PPO",
        "learning_rate": 1e-3,          # Lowered slightly for massive aggregate batch size
        "num_learning_epochs": 5,
        "num_mini_batches": 32,          # Increased from 4 to keep mini-batch sizes reasonable (~49k)
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "clip_param": 0.2,
        "entropy_coef": 0.01,           # Increased slightly for better exploration
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "gamma": 0.99,
        "lam": 0.95,
        "max_grad_norm": 1.0,
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
    },
}
MAX_ITERATIONS = 10000  # total iterations to reach


def latest_ckpt(log_dir):
    files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    nums = [int(re.findall(r"model_(\d+)\.pt", os.path.basename(f))[0]) for f in files]
    return max(nums) if nums else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue training from a checkpoint")
    ap.add_argument("--ckpt", type=int, default=-1, help="-1 = latest saved checkpoint")
    ap.add_argument("-n", "--num_envs", type=int, default=4096, help="default scaled to 16k envs")
    ap.add_argument("--view", action="store_true", help="show viewer (env 0) while training")
    args = ap.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")
    log_dir = "logs/dp_cart"
    os.makedirs(log_dir, exist_ok=True)

    if args.eval:
        env = DoublePendulumCartEnv(1, ENV_CFG, show_viewer=True)
        runner = OnPolicyRunner(env, TRAIN_CFG, log_dir, device=gs.device)
        ck = args.ckpt if args.ckpt >= 0 else latest_ckpt(log_dir)
        if ck is None:
            raise SystemExit(f"No checkpoints in {log_dir} yet. Train first.")
        
        print("Loading checkpoint:", ck)
        runner.load(os.path.join(log_dir, f"model_{ck}.pt"))
        policy = runner.get_inference_policy(device=gs.device)
        
        obs = env.reset()[0] if hasattr(env, "reset") else env.get_observations()
        with torch.inference_mode():
            while True:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if dones.any():
                    print("Episode reset (timeout or cart hit x_limit)")
    else:
        env = DoublePendulumCartEnv(args.num_envs, ENV_CFG, show_viewer=args.view)
        runner = OnPolicyRunner(env, TRAIN_CFG, log_dir, device=gs.device)
        
        if args.resume:
            ck = args.ckpt if args.ckpt >= 0 else latest_ckpt(log_dir)
            if ck is None:
                print("No checkpoint found, starting from scratch.")
            else:
                runner.load(os.path.join(log_dir, f"model_{ck}.pt"))
                print(f"Resumed from checkpoint {ck}")
                
        done = getattr(runner, "current_learning_iteration", 0)
        print(f"Starting at iteration {done}, training {args.num_envs} envs up to {MAX_ITERATIONS}")
        runner.learn(num_learning_iterations=max(MAX_ITERATIONS - done, 1))