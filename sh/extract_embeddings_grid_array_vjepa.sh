#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=emb_grid_vj
#SBATCH --output=outputs/slurms/%A_%a.out
#SBATCH --qos=qos_nmi
#SBATCH --array=1-40%8

# Sharded V-JEPA 2 grid-embedding extraction: ONE subject per H100. GRID-ONLY
# (V-JEPA 2 has no frozen cls; its alignment rep is a trainable pool over the
# grid, recomputed in the model). Reads the 256-frame V-JEPA2 frame grid,
# stores grid (n_slots,2,P=256,d=1024) fp32 uncompressed. ViT-L is bigger than
# DINOv2-base: grid buffer ~82 GB for the largest subject -> mem 160G; ~1.4 TB
# total on disk across 24 video subjects. batch 64 (video forward, T=2 replicate)
# to stay well within the 80 GB H100. Same fp32/none format as the DINOv2 cache.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc '
  module load miniconda
  conda activate cbramod
  SUBJ=$(printf "P%04d" "$SLURM_ARRAY_TASK_ID")
  echo "[array task $SLURM_ARRAY_TASK_ID] V-JEPA2 embedding subject=$SUBJ"
  exec python -m datasets.egobrain_extract_embeddings_grid \
      --data_dir data/EgoBrain \
      --subjects "$SUBJ" \
      --vision_encoder facebook/vjepa2-vitl-fpc64-256 \
      --grid_s 0.2 \
      --dtype float32 \
      --compression none \
      --batch_size 64
'
