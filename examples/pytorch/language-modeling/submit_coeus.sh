#!/bin/bash
#SBATCH --job-name my_job
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --partition gpu
#SBATCH --gres=gpu:3
#conda activate pytorch

srun -o result.out --ntasks=1 -N 1 python run_clm.py \
    --model_name_or_path openai-community/gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm