#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:15:00
#SBATCH --job-name=hand_dbg
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi
module load miniconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wilor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Re-run the REAL viz path under faulthandler to confirm the cv2-thread fix
# (or trace the exact crash if it persists).
python -X faulthandler -u datasets/egobrain_hand_labels_viz.py \
    --data_dir data/EgoBrain --subject P0001 --n_examples 3 \
    --out_dir outputs/hand_label_viz --device cuda --dtype float16
