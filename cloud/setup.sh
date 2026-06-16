#!/usr/bin/env bash
# ProMerge cloud bootstrap — Linux GPU box (GCP / AWS Deep Learning VM, Ubuntu 20.04/22.04).
# Sets up: system EGL/MuJoCo libs, a Python venv, project deps, and the LIBERO env
# (robosuite + bddl) so real LIBERO-Spatial training AND success-rate eval can run headless.
#
# Usage:
#   cd ProMerge && bash cloud/setup.sh
#   source .venv/bin/activate && source cloud/env.sh
set -euo pipefail

echo "==> [1/5] System packages (EGL / MuJoCo / robosuite headless rendering)"
sudo apt-get update -y
sudo apt-get install -y \
  python3.10 python3.10-venv python3-pip git wget unzip \
  libgl1-mesa-glx libgl1-mesa-dev libglew-dev libosmesa6-dev \
  libegl1 libgles2 libglfw3 patchelf ffmpeg

echo "==> [2/5] Python venv"
python3.10 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel setuptools

echo "==> [3/5] Project deps (torch grabs the CUDA wheel automatically on Linux)"
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

echo "==> [4/5] LIBERO env (robosuite + bddl) for training data + official eval"
mkdir -p external
if [ ! -d external/LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
fi
# Install LIBERO's deps then the package (editable). If a pin clashes with our torch,
# our CUDA torch should remain; re-run `pip install -r requirements.txt` if needed.
pip install -r external/LIBERO/requirements.txt || true
pip install -e external/LIBERO

echo "==> [5/5] Writing cloud/env.sh (source it each session)"
cat > cloud/env.sh <<'EOF'
# Source after activating the venv: `source cloud/env.sh`
export MUJOCO_GL=egl            # headless offscreen rendering for MuJoCo/robosuite
export DATALOADER_WORKERS=8     # data-loading parallelism on the cloud box
export PYOPENGL_PLATFORM=egl
# export WANDB_API_KEY=...       # set your key (or run `wandb login`)
EOF

echo ""
echo "✅ Setup done."
echo "Next:"
echo "  source .venv/bin/activate && source cloud/env.sh"
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='yifengzhu-hf/LIBERO-datasets', repo_type='dataset', local_dir='data', allow_patterns='libero_spatial/*')\""
echo "  # sanity-check rendering:"
echo "  python -c \"import os; os.environ['MUJOCO_GL']='egl'; from libero.libero import benchmark; print('LIBERO OK', list(benchmark.get_benchmark_dict()))\""
