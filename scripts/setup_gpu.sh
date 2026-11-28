#!/usr/bin/env bash
# One-shot setup of a fresh Linux GPU machine for MJX training (single_leg_demo/MJX.md).
#
#   bash setup_gpu.sh [BRANCH]          # default branch: single-leg-mjx
#
# Before running: copy the leg model (generated from Onshape, not in git) next to this script:
#   rsync -a urdf/gradus/demo/ SERVER:~/gradus-demo-model/
# The repo is private: have a GitHub token or SSH key on the server that can read it.
set -euo pipefail
BRANCH="${1:-single-leg-mjx}"
REPO="${REPO:-https://github.com/softrobotrl/gradus-simulation.git}"
DIR="${DIR:-$HOME/gradus-simulation}"
MODEL="${MODEL:-$HOME/gradus-demo-model}"

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no NVIDIA GPU visible"; exit 1; }
echo "== CPU cores: $(nproc)"

if [ ! -d "$DIR/.git" ]; then git clone "$REPO" "$DIR"; fi
cd "$DIR" && git fetch origin && git checkout "$BRANCH" && git pull --ff-only origin "$BRANCH"

[ -f "$MODEL/rig.xml" ] || { echo "leg model missing: copy urdf/gradus/demo/ to $MODEL first"; exit 1; }
mkdir -p urdf/gradus && rm -rf urdf/gradus/demo && cp -r "$MODEL" urdf/gradus/demo

python3 -m venv "$HOME/.venvs/gradus-mjx"
. "$HOME/.venvs/gradus-mjx/bin/activate"
pip install -q --upgrade pip
# Versions as tested on the laptop (JAX 0.10 hangs running MJX; brax 0.13 matches JAX 0.6.2).
pip install -q "jax[cuda12]==0.6.2" mujoco==3.14.0 mujoco-mjx==3.14.0 brax==0.13.0 flax==0.11.2 \
    "numpy<2.3" tensordict==0.10.0 rsl-rl-lib==5.5.1 imageio pillow scipy \
    torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cpu

python -c "import jax; d = jax.devices(); print('JAX devices:', d); assert d[0].platform == 'gpu', 'JAX is not using the GPU'"
echo "== MJX vs CPU agreement test"
PYTHONPATH=. python -m unittest tests.test_mjx_env
echo "== ready. Train with e.g.:"
echo "PYTHONPATH=. python -m single_leg_demo.train_mjx --run-name mjx_aligned --dr --rail-friction 3 \\"
echo "    --cmd-min 0.05 --cmd-max 0.08 --cmd-period 3 --set cmd_period_jitter=0.333 \\"
echo "    --set servo_layout=single_x --set leg_yaw_deg=45 --episode-steps 600 --num-envs 4096 --timesteps 200e6"
