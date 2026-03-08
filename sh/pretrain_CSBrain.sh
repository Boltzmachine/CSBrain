#!/bin/bash

#!/bin/zsh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=csbrain
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi


# python pretrain_main.py \
#     --model CSBrain \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --run_name CSBrain2

# python pretrain_main.py \
#     --model OurModel \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --run_name ours_all_zerosync_16

python pretrain_main.py \
    --model Align \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --model_dir outputs/ \
    --dataset_dir mix \
    --run_name align

# python pretrain_main.py \
#     --model LLMVQ \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --run_name llm_vq
