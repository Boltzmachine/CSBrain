#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=400G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=csbrain
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

echo "Job started at $(date)"

echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

python pretrain_main.py \
    --model CSBrain \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --model_dir outputs/pretrain_CSBrain \
    --dataset_dir data/tueg.db

echo "Job completed at $(date)"

