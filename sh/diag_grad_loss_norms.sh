#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --job-name=gradnorm
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Per-loss gradient-magnitude diagnostic. Loads a WorldModel pretrain
# checkpoint, replicates the exact training-step loss assembly on a few real
# EgoBrain batches, and reports the grad norm each loss term contributes to the
# encoder vs all trainable params. CKPT overridable from the env.

set -euo pipefail
module load miniconda
CKPT="${CKPT:-outputs/wm-dino-dense/epoch20_loss2.1313838958740234.pth}"

conda run -n cbramod python diag_grad_loss_norms.py \
    --ckpt "$CKPT" \
    --n_batches 6 \
    --batch_size 24 \
    --num_workers 6
