# Training on a GPU with MJX

**MJX** is MuJoCo rewritten in JAX, so thousands of copies of the leg can step
together on a GPU. **Brax** provides the PPO trainer that runs on it. The CPU
environment ([rig_env.py](rig_env.py)) stays the reference; the MJX version
([mjx_env.py](mjx_env.py)) copies it, and [tests/test_mjx_env.py](../tests/test_mjx_env.py)
checks they agree.

## How the MJX version relates to the CPU one

| | CPU (`train.py`) | MJX (`train_mjx.py`) |
|---|---|---|
| Simulator | MuJoCo, 64 envs on a few CPU threads | MJX, thousands of envs on one GPU |
| Trainer | rsl-rl PPO (PyTorch) | Brax PPO (JAX) |
| Reward | [rewards.py](rewards.py) | same terms, mirrored in [mjx_rewards.py](mjx_rewards.py) |
| Episode starts | reset + randomize each episode | sampled from a bank of CPU-env resets (`build_reset_bank`), so randomized physics is still redrawn every episode |
| Clone / KL anchor, warm-up, randomization ramp | yes | not yet |
| Output | `model_N.pt` | `policy.npz` (plain arrays; runs on the CPU env without JAX) |

Measured agreement over 60 steps from the same start (test_mjx_env): positions
within 5e-8 m, rewards within 4e-5, actor inputs within 3e-8 except the
accelerometer (within 0.01 g, the size of the sensor noise added in training).

## Set up a GPU machine

1. Clone the repo and check out `single-leg-mjx`.
2. Copy the leg model: it is generated from Onshape and git-ignored.
   `rsync -a urdf/gradus/demo/ SERVER:REPO/urdf/gradus/demo/`
3. Create the environment from `environment.yml` (it pins `jax[cuda12]==0.6.2`;
   JAX 0.10 hangs running MJX), then add Brax:
   `pip install brax==0.13.0 flax==0.11.2`
4. Check the GPU is used: `python -c "import jax; print(jax.devices())"` should list a CUDA device.
5. Check agreement: `python -m unittest tests.test_mjx_env`

## Train and judge

```
python -m single_leg_demo.train_mjx --run-name mjx_v5 --dr --rail-friction 3 \
    --cmd-min 0.05 --cmd-max 0.08 --cmd-period 3 --set cmd_period_jitter=0.333 \
    --episode-steps 600 --num-envs 4096 --timesteps 200e6
python -m single_leg_demo.reward_check.passes --period 3 --dr 4 logs/single_leg_demo/mjx_v5/policy.npz
```

Progress prints every `--timesteps / --evals` steps and is saved to
`RUN/progress.jsonl`; `--wandb PROJECT` also logs to Weights & Biases.
