#!/bin/bash -l
#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 1           
#SBATCH -p shared-gpu
#SBATCH -C gpu_count:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
module load miniconda3

# Enable Python fault handler
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning' CUDA error checking
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

source activate /vast/home/ajherman/miniconda3/envs/transformer
#pip install datasets
#export PATH="/vast/home/ajherman/miniconda3/envs/transformer/bin:$PATH"

# srun -o original.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 gpt2_train.py --output_dir original --num_train_epochs 2 --config_file medium.json --per_device_train_batch_size 13 --mixed_precision --save_steps 1000 --logging_steps 50 --eval_steps 50 --gradient_accumulation_steps 10 --max_step 1000000 --load_from_checkpoint &

#srun -o relu.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 gpt2_train.py --output_dir relu --num_train_epochs 2 --config_file relu.json --per_device_train_batch_size 13 --mixed_precision --save_steps 1000 --logging_steps 50 --eval_steps 50 --gradient_accumulation_steps 10 --max_steps 1000000 --load_from_checkpoint &

srun -o original.out --ntasks=1 -N 1 torchrun --nproc_per_node 4 gpt2_train.py --output_dir original --num_train_epochs 1 --per_device_train_batch_size 13 --mixed_precision --logging_steps 50 --eval_steps 50 --max_step 1000000 --no_train --learning_rate 0.0 &



#nvidia-smi

#srun -o tiny_test.out --ntasks=1 -N 1 torchrun --nproc_per_node 2 gpt2_train.py --output_dir /tmp/test-clm --num_train_epochs 100 --config_file config.json --per_device_train_batch_size 12 --mixed_precision --save_steps 2000 &
