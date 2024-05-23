#!/bin/bash -l
#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 1           
#SBATCH -p shared-gpu
module load miniconda3
source activate /vast/home/ajherman/miniconda3/envs/transformer
#pip install datasets
#export PATH="/vast/home/ajherman/miniconda3/envs/transformer/bin:$PATH"

srun -o tiny_test.out --ntasks=1 -N 1 python gpt2_train.py --output_dir /tmp/test-clm --num_train_epochs 10 --config_file tiny.json --per_device_train_batch_size 8 &

