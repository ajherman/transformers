#!/bin/bash -l
#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 1           
#SBATCH --mem=40G  
#SBATCH -p shared-gpu
##SBATCH -C gpu1_memory:40GB
module load miniconda3
source activate /vast/home/ajherman/miniconda3/envs/transformer
#pip install datasets
#export PATH="/vast/home/ajherman/miniconda3/envs/transformer/bin:$PATH"

srun -o result.out --ntasks=1 -N 1 python run_clm.py \
    --model_type gpt2 \
    --tokenizer_name openai-community/gpt2 \
    --config_overrides "n_layer=10,n_head=6" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm

# --model_type gpt2 --tokenizer_name openai-community/gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir result_dir

