#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-finetune
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)

LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"


# python finetune_main.py \
#     --model Align \
#     --downstream_dataset PhysioNet-MI \
#     --datasets_dir data/preprocessed/physionet_mi \
#     --num_of_classes 4 \
#     --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
#     --foundation_dir outputs/Align-alljoined-mask/epoch22_loss4.793579578399658.pth \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --use_pretrained_weights \
#     --in_dim 40 \
#     --out_dim 40 \
#     --d_model 40 \
#     --seq_len 20 \
#     --nhead 4 \
#     --n_layer 24 \
#     --dropout 0.3 \
#     --weight_decay  0.01 \
#     --lr 0.00005 

FOUNDATION_DIR="${1:-outputs/wm-mix-egobrain-bands2/epoch20_loss0.7221527695655823.pth}"
SEED="${2:-42}"
NUM_WORKERS="${3:-4}"
SEGMENT_INDEX="${4:-0}"
HIGHPASS_HZ="${5:-0}"
SYM_PROB="${6:-0.25}"   # 思路1 symmetrize-aug probability
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
# Current config: 思路1 ONLY (--symmetrize_aug) for a clean A/B vs the no-aug
# baseline. The lateralization-flip aug + TTA are left commented below (they
# target the already-solved left/right axis and slightly hurt). Tag the
# run/results so tracks don't collide in wandb / the CSV.
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}_hp${HIGHPASS_HZ}_symp${SYM_PROB}"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv "outputs/finetune_results_symp${SYM_PROB}.csv" \
    --use_pretrained_weights \
    --use_initial_segment_only \
    --segment_index "$SEGMENT_INDEX" \
    --highpass_hz "$HIGHPASS_HZ" \
    --num_workers "$NUM_WORKERS" \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.00005 \
    --symmetrize_aug \
    --symmetrize_aug_prob "$SYM_PROB"
    # 思路1: synthesize 'both fists' from single-hand trials via the symmetric
    # projection x_sym = 0.5*(x + flip(x)) (raw channel swap), relabeled to
    # --symmetrize_target_label (default 2). Targets the bilateral-extent
    # confusion (both fists <-> single fist); needs no pretrained split.
    #
    # To re-enable the lateralization-flip aug / TTA instead (or in addition),
    # add: --lateral_flip_aug --lateral_flip_prob 0.5 --lateral_flip_tta
    # (those require a --lateralization_flip-pretrained FOUNDATION_DIR).
    # --lateral_flip_aug reuses the pretrained x_bi/x_lat split (so FOUNDATION_DIR
    # must be a --lateralization_flip checkpoint) to augment with
    # x_flip = x_bi + flip(x_lat) and remap labels (--flip_label_map "1,0,2,3":
    # left fist<->right fist; both fists / both feet unchanged).

# VISION_ENCODER="facebook/dinov2-base"   # swap to DINOv3 id once access is granted
# IMAGE_MODE="raw"                        # 'raw' matches the paper; 'spectrogram' = STFT grid
# WANDB_RUN_NAME="dinov3eeg_$(basename ${VISION_ENCODER})_${IMAGE_MODE}_lora"

# python finetune_main.py \
#     --model DINOv3EEG \
#     --downstream_dataset PhysioNet-MI \
#     --datasets_dir data/preprocessed/physionet_mi \
#     --num_of_classes 4 \
#     --model_dir outputs/CSBrain/finetune_DINOv3EEG_PhysioNet \
#     --vision_encoder "$VISION_ENCODER" \
#     --image_mode "$IMAGE_MODE" \
#     --use_lora \
#     --lora_rank 256 \
#     --lora_alpha 32 \
#     --wandb_run_name "$WANDB_RUN_NAME" \
#     --results_csv outputs/finetune_results.csv \
#     --dropout 0.1 \
#     --weight_decay 0.01 \
#     --lr 0.0001

wait

# 记录任务完成时间到日志文件
echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"


