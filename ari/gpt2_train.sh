#!/bin/bash -l
#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 1           
#SBATCH --mem=0 
#SBATCH -p shared-gpu-ampere
#SBATCH -C gpu_count:2
#SBATCH --exclusive
#SBATCH --cpus_per_task=16
module load miniconda3
source activate /vast/home/ajherman/miniconda3/envs/transformer
#pip install datasets
#export PATH="/vast/home/ajherman/miniconda3/envs/transformer/bin:$PATH"


srun -o result.out --ntasks=1 -N 1 torchrun --nproc_per_node 2 gpt2_train.py --output_dir checkpoints_full_size --num_train_epochs 100 --config_file config.json --per_device_train_batch_size 16 --mixed_precision --save_steps 2000 &
