#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --job-name=split-gsym
#SBATCH --output=outputs/slurms/%j.out

# CORRECTED antisymmetry-selectivity probe.
#
# The raw selectivity R(z_lat)-R(z_bi) is SELF-CONFOUNDED: writing gbar=(g+Sg)/2 and
# delta=(g-Sg)/2, we have  g*z - S(g*z) = gbar*(z-Sz) + delta*(z+Sz).  The symmetric
# signal (z+Sz) carries ~13x the energy of the antisymmetric one, so a merely LOPSIDED
# gate fabricates antisymmetry out of bilateral signal: on a signal with ZERO lateralized
# content, sel = delta^2/(delta^2+gbar^2) - delta^2/(delta^2+(1-gbar)^2) > 0 for every
# gbar < 0.5 -- which every gate in this family satisfies.
# The random-init "null = 0" is circular: its gate is a global SCALAR (gate_head.weight
# is init'd to zeros), the one class with delta == 0.
#
# selectivity_gsym symmetrises the gate (keeping its channel topography, feature
# structure and data dependence) and asks whether it selects lateralized SIGNAL beyond
# what its own lopsidedness mechanically produces.
# NOTE: conda must be active BEFORE sbatch.

set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs/split_probe

for RUN in wm-new-gradneg wm-new wm-new-biggradneg wm-new-alpha1; do
  for EP in 8 9 10; do
    CK=$(ls outputs/$RUN/epoch${EP}_loss*.pth 2>/dev/null | head -1)
    [ -z "$CK" ] && continue
    TAG="gs_${RUN}_ep${EP}"
    echo "=========== $TAG ==========="
    python -u scripts/probe_bilateral_split.py --ckpt "$CK" --tag "$TAG" \
        --domain egobrain --stages antisym --max_batches 40 --ego_clips 4000 \
        --num_workers 0 --out "outputs/split_probe/${TAG}.json" || echo "FAILED $TAG"
  done
done
# scalar-gate control (random init): delta == 0 by construction
python -u scripts/probe_bilateral_split.py \
    --ckpt outputs/wm-new-gradneg/epoch10_loss2.3018484115600586.pth --tag gs_randinit \
    --random_init --domain egobrain --stages antisym --max_batches 40 --ego_clips 4000 \
    --num_workers 0 --out outputs/split_probe/gs_randinit.json
echo ALL DONE
