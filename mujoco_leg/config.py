"""Configuration for the tendon-leg tracer PPO."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # model
    model_path: str = "urdf/gradus/tracer/leg.xml"
    tip_site: str = "tip"

    # environment
    n_envs: int = 8192
    n_steps: int = 32
    n_substeps: int = 5          # physics steps per control step (0.002 s each)
    episode_steps: int = 600     # 6 s at 100 Hz
    shape_points: int = 128
    shape_radius: float = 0.035  # metres
    lookahead: int = 8
    lookahead_dt: float = 0.025  # seconds between look-ahead samples
    period: float = 4.0          # seconds for one full shape
    init_qpos_noise: float = 0.004
    init_qvel_noise: float = 0.004
    warmup_steps: int = 0
    track_sigma: float = 0.008   # metres, Gaussian tracking reward width

    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 8
    entropy: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256

    # run
    n_iters: int = 200
    minutes: float = 55.0
    seed: int = 0
    log_dir: str = "logs/mujoco_leg"
    save_every: int = 10
    run_name: str = ""
