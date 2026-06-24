#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
# Modest GPU is plenty for WiLoR; l40s/a40 schedule far sooner than h100.
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --job-name=hand_viz
#SBATCH --output=outputs/slurms/%j.out
# qos_nmi grants l40s=12 (qos_ying_rex grants a40 but l40s=0). Match --gres.
#SBATCH --qos=qos_nmi

# Render annotated montage examples for the EgoBrain hand-label pipeline so the
# labels can be eyeballed. Short walltime so it backfills onto a free GPU fast.
# WiLoR is too slow on a login-node CPU, hence the GPU job.
#   sbatch sh/egobrain_hand_labels_viz.sh                # P0001, 12 examples
#   SUBJECT=P0003 N=16 sbatch sh/egobrain_hand_labels_viz.sh
# Montages: outputs/hand_label_viz/<subject>/clipXXXX_winI_<label>.png

module load miniconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wilor
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SUBJECT="${SUBJECT:-P0001}"
N="${N:-12}"

python datasets/egobrain_hand_labels_viz.py \
    --data_dir data/EgoBrain \
    --subject "${SUBJECT}" \
    --n_examples "${N}" \
    --out_dir outputs/hand_label_viz \
    --window_s 1.0 --stride_s 1.0 --clip_s 4.0 --fs_out 200 \
    --erp_latency_s 0.5 --frames_per_window 7 \
    --move_thresh 0.15 --min_det_frac 0.4 --handedness_source hybrid \
    --max_frame_width 1280 --device cuda --dtype float16
