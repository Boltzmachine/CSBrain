#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --job-name=split-eb
#SBATCH --output=outputs/slurms/%j.out

# Error bars for the headline antisymmetry-SELECTIVITY statistic.
#
# Pretraining in this repo is not run-to-run reproducible and every config has
# exactly ONE seed, so cross-run p-values are not available. The defensible error
# bar is the WITHIN-RUN spread over the last epochs (ep8/9/10) plus the per-batch
# SE inside each. Anything smaller than that spread is noise.
#
# selectivity = R(z_lat) - R(z_bi), where R(u) = ||u-Pu||^2 / (||u-Pu||^2+||u+Pu||^2).
# It is INVARIANT to a scalar rescale of the gate, and its null for ANY constant
# gate is exactly 0 (verified: the random-init ckpt returns -0.0000000).
#
# NOTE: conda must be active BEFORE sbatch.

set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/split_probe
mkdir -p "$OUT"

for RUN in wm-new-gradneg wm-new wm-new-biggradneg wm-new-alpha1; do
  for EP in 8 9 10; do
    CK=$(ls outputs/$RUN/epoch${EP}_loss*.pth 2>/dev/null | head -1)
    [ -z "$CK" ] && { echo "skip $RUN ep$EP"; continue; }
    TAG="eb_${RUN}_ep${EP}"
    echo "=================== $TAG ==================="
    python -u scripts/probe_bilateral_split.py --ckpt "$CK" --tag "$TAG" \
        --domain egobrain --stages antisym,eqconst \
        --max_batches 40 --ego_clips 4000 --num_workers 0 \
        --out "$OUT/${TAG}.json" || echo "FAILED: $TAG"
  done
done

echo "ALL DONE"
