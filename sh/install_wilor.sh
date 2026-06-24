#!/bin/bash
# Build a DEDICATED conda env ("wilor") for deriving EgoBrain hand labels with
# WiLoR. Kept separate from the cbramod training env on purpose: WiLoR needs
# chumpy (to unpickle MANO_RIGHT.pkl), and chumpy only builds/imports on
# numpy<1.24 + python<3.11 — incompatible with cbramod (numpy 2.x / py3.10 ok
# but numpy 2 breaks chumpy). The label pipeline only PRODUCES data (HDF5),
# which cbramod reads back with h5py, so the envs never need to coincide.
#
# RUN ON A LOGIN NODE (needs internet). ~a few minutes (downloads torch etc.).
#   bash sh/install_wilor.sh
# Model weights auto-download from HuggingFace on first predict().
set -euo pipefail

ENV=wilor
module load miniconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env list | grep -qE "/${ENV}$" || conda create -n "$ENV" python=3.10 -y
conda activate "$ENV"
pip install -U pip setuptools wheel

# Pin numpy<1.24 FIRST so chumpy can build (it uses np.bool/np.float, removed
# in 1.24) and import (python 3.10 still has inspect.getargspec, gone in 3.11).
pip install "numpy==1.23.5"
# chumpy's setup.py does `import pip`, which fails under pip build isolation —
# --no-build-isolation lets it build against the env's pip + numpy.
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"

# torch<=2.5 (WiLoR's pin); default PyPI wheels are CUDA-enabled on linux.
pip install torch==2.5.0 torchvision==0.20.0
pip install "numpy==1.23.5"                 # re-pin (torch may have bumped it)

# WiLoR-mini itself without dep resolution, then its deps at numpy-safe pins.
pip install --no-deps git+https://github.com/warmshao/WiLoR-mini
pip install smplx==0.1.28 timm einops ultralytics==8.1.34 \
    huggingface_hub "scikit-image==0.21.0" roma decord h5py tqdm dill
# opencv: ultralytics pulls the latest opencv-python, which needs numpy>=2 and
# so won't import on our numpy 1.23. Force a numpy-1-compatible headless build
# (it provides the same cv2 module ultralytics imports).
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install "opencv-python-headless==4.9.0.80"
pip install "numpy==1.23.5"                 # final re-pin

echo "[install] versions:"
python -c "import numpy,torch,torchvision; print(' numpy',numpy.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
echo "[install] import + chumpy check:"
python - <<'PY'
import importlib, numpy as np, torch
for m in ('chumpy', 'cv2', 'decord', 'h5py', 'ultralytics', 'smplx', 'dill',
          'wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline'):
    importlib.import_module(m)
    print('  ok', m)
# Warm the weight cache NOW (on this online login node) so offline GPU compute
# nodes find MANO/WiLoR/YOLO checkpoints under site-packages/wilor_mini.
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
pipe = WiLorHandPose3dEstimationPipeline(device=torch.device('cpu'), dtype=torch.float32)
pipe.predict((np.random.rand(256, 256, 3) * 255).astype('uint8'))
print("wilor env ready; model weights cached in the env.")
PY
