import torch

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Mixed precision training requires CUDA-enabled GPUs.")

import os 

os.environ["HF_DATASETS_OFFLINE"] = "0"

from datasets import load_dataset

dataset_name = "monology/pile-uncopyrighted"
train_dataset = load_dataset(dataset_name, split='train', streaming=True, block_size=655360)