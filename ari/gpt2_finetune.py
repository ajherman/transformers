import torch
from torch.optim import Adam
import torch.distributed as dist
import traceback

#Ensure CUDA is available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Mixed precision training requires CUDA-enabled GPUs.")

import sys
import argparse
import csv
import os 
import numpy as np
import time
from tqdm import tqdm

os.environ["HF_DATASETS_OFFLINE"] = "0"

# Parse command line arguments
parser = argparse.ArgumentParser(description='GPT-2 Training')
parser.add_argument('--output_dir', type=str, default='./gpt2-wikitext', help='Output directory')
parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--save_steps', type=int, default=10000, help='Number of updates steps before checkpoint saves')
parser.add_argument('--save_total_limit', type=int, default=2, help='Limit the total amount of checkpoints and deletes the older checkpoints')
parser.add_argument('--use_local_transformers', action='store_true', help='Use local transformers repository')
# parser.add_argument('--config_file', type=str, default=None, help='Config file')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
parser.add_argument('--context_length', type=int, default=256, help='Context length')
parser.add_argument('--load_from_checkpoint', action='store_true', help='Load from checkpoint')
parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of gradient accumulation steps')
parser.add_argument('--logging_steps', type=int, default=2500, help='Number of steps between logs')
parser.add_argument('--eval_steps', type=int, default=1000, help='Number of steps before evaluation')
parser.add_argument('--max_steps', type=int, default=100000, help='Maximum number of steps')
parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Learning rate')
args = parser.parse_args()

# Path to the 'src' directory of your local transformers repository
use_local_transformers = args.use_local_transformers
if use_local_transformers:
    # Path to the 'src' directory of your local transformers repository
    path_to_transformers = '../src/transformers'

    # Prepend this path to sys.path
    if path_to_transformers not in sys.path:
        sys.path.insert(0, path_to_transformers)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, AutoModelForCausalLM, GPT2Config
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerControl, get_cosine_schedule_with_warmup
from datasets import load_dataset

import logging
from transformers.activations import ACT2FN


tic,toc = 0.0,time.time()

try:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='nccl')

    # Read in config file
    # config = GPT2Config.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token # Add pad token

    # model = ModifiedGPTModel(config=config) 
    
    # Create an modify model
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    for block in model.transformer.h:
        block.mlp.act = ACT2FN['relu']

    # print("Config:\n", model.config)

    # Print number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of model parameters:", num_params)

    # Load the dataset
    #logging.getLogger("datasets").setLevel(logging.DEBUG)

    dataset_name = "monology/pile-uncopyrighted"
    train_dataset = load_dataset(dataset_name, split='train', streaming=True) #, block_size=655360)
    eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')

    print("Finished loading datasets")

    # Tokenize the dataset 
    def encode(examples):

        tokenized_texts = tokenizer(examples['text'], truncation=True, max_length=model.config.n_ctx, padding='max_length')
        return tokenized_texts

    train_dataset = train_dataset.map(encode, batched=True)
    eval_dataset = eval_dataset.map(encode, batched=True)

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir, # output directory
        overwrite_output_dir=True, # overwrite the content of the output directory
        per_device_train_batch_size=args.per_device_train_batch_size, # batch size for training
        save_steps=args.save_steps, # number of updates steps before checkpoint saves
        logging_steps= args.logging_steps, # Number of steps between logs
        seed=args.seed, # Random seed
        save_total_limit=args.save_total_limit, # limit the total amount of checkpoints and deletes the older checkpoints
        eval_strategy="steps", # evaluation strategy to adopt during training
        fp16=args.mixed_precision, # Mix percision
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        eval_steps=args.eval_steps, # number of steps before evaluation
        warmup_steps=2000, # number of warmup steps for learning rate scheduler
        learning_rate=args.learning_rate, # learning rate
        lr_scheduler_type='cosine', # learning rate scheduler type
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        weight_decay=0.01, 
    )

    # Use this to periodically trigger events during training
    class CustomCallback(TrainerCallback):
        def __init__(self, trainer,tokenizer):
            self.trainer = trainer
            self.tokenizer = tokenizer

        def on_epoch_end(self, args, state, control, **kwargs):
            # Perform custom evaluation at the end of each epoch
            global tic,toc
            if args.evaluation_strategy == "epoch":
                pass
                #self.custom_evaluation()
            tic,toc = toc,time.time()
            print("Time per epoch: ",(toc-tic)/60,"m")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None:
                print(f"Evaluation results at step {state.global_step}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

            #### Directly compute from model ####
            
            model = self.trainer.model
            tokenizer = self.tokenizer
            device = model.device
            
            model.eval()

            test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            # test = test.shuffle()
            encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

            max_length = model.config.n_positions
            stride = model.config.n_positions 
            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end_loc = 0
            for begin_loc in tqdm(range(0, seq_len, stride)):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            ppl = torch.exp(torch.stack(nlls).mean())
            if local_rank==0:
                print("Perplexity (single string version): ",ppl)

            model.train()

            ###########################################
        
            with open(args.output_dir+'/metrics.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([state.epoch, ppl.item()])

        def on_log(self, args, state, control, **kwargs):
            pass


# # Create the optimizer and scheduler
# def get_optimizer_and_scheduler(model, num_warmup_steps, num_training_steps):
#     optimizer = Adam(model.parameters(), lr=2.5e-4, weight_decay=0.01)
#     scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
#     return optimizer, scheduler

# optimizer, scheduler = get_optimizer_and_scheduler(model, warmup_steps, total_training_steps)

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.add_callback(CustomCallback(trainer,tokenizer))    
    
    # Evaluate model before fine-tuning
    eval_results = trainer.evaluate()
    print("Evaluation results:\n", eval_results)

    # Perplexity
    perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
    print("Perplexity:", perplexity)

    # Fine-tune model
    trainer.train(resume_from_checkpoint=args.load_from_checkpoint) # More precise version would be to pass args.checkpoint_dir explicitly

    # Save model
    model.save_pretrained(args.output_dir)
    print("Model saved")

    # Evaluate model
    eval_results = trainer.evaluate()
    print("Evaluation results:\n", eval_results)

    # Perplexity
    perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
    print("Perplexity:", perplexity)

    toc = time.time()
    print("Duration: ",(toc-tic)/60,"m")
except Exception as e:
    # Print the full traceback
    print("Exception occurred:", e)
    traceback.print_exc()
