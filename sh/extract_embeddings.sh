#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=emb_extract
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Pre-compute the frozen DINOv2 embeddings of the EgoBrain frames so world-model
# pretraining can load them (--use_cached_embeddings in sh/pretrain_worldmodel.sh)
# instead of running the encoder on the fly. Reads the existing uint8 frame cache
# (cache_frames_<enc>_w..), runs the encoder once per (clip x window) in BOTH
# orientations, and writes cls/cls_flip + grid/grid_flip (float16) to
# cache_embeddings_<enc>_w.. — ~53 GB for dinov2-base (24 subjects, 66,914 frame
# slots). The window/stride/erp/n_windows/sz/--vision_encoder MUST match the run
# (and the frame cache), or the slug won't line up.
#
# Self-activates conda inside a zsh LOGIN shell (the pattern proven to work in
# this environment; see feedback_shell_and_remote / project_sbatch_env_inherit).
# A plain `#!/bin/bash` + `conda activate` trips conda's activate.d scripts under
# `set -u` (ADDR2LINE unbound) and the bash hook, so do NOT reintroduce that.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc 'module load miniconda; conda activate cbramod; exec python -m datasets.egobrain_extract_embeddings \
    --data_dir data/EgoBrain \
    --subjects all \
    --vision_encoder facebook/dinov2-base \
    --window_s 1.0 --stride_s 1.0 --erp_latency_s 0.5 \
    --n_windows 2 --fs_out 200 --batch_size 256'
