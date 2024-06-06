import torch
import torch.distributed as dist
import traceback

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Mixed precision training requires CUDA-enabled GPUs.")

import sys
import argparse
import csv
import os 
import numpy as np
import time

os.environ["HF_DATASETS_OFFLINE"] = "0"

# Parse command line arguments
parser = argparse.ArgumentParser(description='GPT-2 Training')
parser.add_argument('--output_dir', type=str, default='./gpt2-wikitext', help='Output directory')
parser.add_argument('--num_train_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--save_steps', type=int, default=10000, help='Number of updates steps before checkpoint saves')
parser.add_argument('--save_total_limit', type=int, default=2, help='Limit the total amount of checkpoints and deletes the older checkpoints')
parser.add_argument('--use_local_transformers', action='store_true', help='Use local transformers repository')
parser.add_argument('--config_file', type=str, default='config.json', help='Config file')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
parser.add_argument('--context_length', type=int, default=256, help='Context length')
parser.add_argument('--load_from_checkpoint', action='store_true', help='Load from checkpoint')
parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of gradient accumulation steps')
parser.add_argument('--logging_steps', type=int, default=2500, help='Number of steps between logs')
args = parser.parse_args()

# Path to the 'src' directory of your local transformers repository
use_local_transformers = args.use_local_transformers
if use_local_transformers:
    # Path to the 'src' directory of your local transformers repository
    path_to_transformers = '../transformers/src'

    # Prepend this path to sys.path
    if path_to_transformers not in sys.path:
        sys.path.insert(0, path_to_transformers)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, AutoModelForCausalLM, GPT2Config
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerControl
from datasets import load_dataset
import logging

tic = time.time()

try:
    dist.init_process_group(backend='nccl')


    # Read in config file
    config = GPT2Config.from_json_file(args.config_file)

    # Load pre-trained model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Add pad token
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

    print("Config:\n", model.config)

    # Print number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of model parameters:", num_params)

    # Load the dataset
    logging.getLogger("datasets").setLevel(logging.DEBUG)


    dataset_name = "EleutherAI/the_pile"
    # train_dataset = load_dataset(dataset_name, subsets = ['hacker_news', 'enron_emails'])
    train_dataset = load_dataset(dataset_name, split='train')
    eval_dataset = load_dataset(dataset_name, subsets = ['reddit'])

    # train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')

    print("Finished loading datasets")

    # Tokenize the dataset
    def encode(examples):
        # tokens = tokenizer(example['text'])
        # tokens = tokens[:max_len]  # Truncate to 100 tokens
        # example['text'] = tokenizer.convert_tokens_to_string(tokens)
        
        # # Tokenize each string in the 'text' field
        # tokenized_texts = [tokenizer.tokenize(text)[:max_len] for text in example['text']]
        # # Convert the tokens back to strings
        # example['text'] = [tokenizer.convert_tokens_to_string(tokens) for tokens in tokenized_texts]
        # return example

        tokenized_texts = tokenizer(examples['text'], truncation=True, max_length=model.config.n_ctx, padding='max_length')
        return tokenized_texts

        # return tokenizer(examples['text']) # Previous version

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
        num_train_epochs=args.num_train_epochs, # number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size, # batch size for training
        save_steps=args.save_steps, # number of updates steps before checkpoint saves
        logging_steps= args.logging_steps, # Number of steps between logs
        seed=args.seed, # Random seed
        save_total_limit=args.save_total_limit, # limit the total amount of checkpoints and deletes the older checkpoints
        evaluation_strategy="epoch", # evaluation strategy to adopt during training
        fp16=args.mixed_precision, # Mix percision
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        # eval_steps=1000, # number of steps before evaluation
        warmup_steps=500, # number of warmup steps for learning rate scheduler
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=5000,
        # weight_decay=0.01, 
    )

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     perplexity = torch_exp(torch.tensor(trainer.eval_loss))
    #     metrics = {'perplexity': perplexity.item()}
    #     # write_metrics_to_csv(metrics, 'metrics.csv')

    #     with open('metrics.csv', 'a', newline='') as file:
    #         writer = csv.writer(file)
    #         for key, value in metrics.items():
    #             writer.writerow([training_args.global_step, key, value])

    #     return metrics

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     # predictions, labels = torch.from_numpy(predictions), torch.from_numpy(labels)
    #     # predictions = predictions.view(-1,predictions.shape[-1])
    #     # labels = labels.view(-1)
    #     # loss = torch.nn.functional.cross_entropy(predictions, labels)
    

    #     # Ensure predictions and labels are tensors
    #     predictions = torch.tensor(predictions)
    #     labels = torch.tensor(labels)
        
    #     # Reshape predictions and labels to be compatible with CrossEntropyLoss
    #     predictions = predictions.view(-1, predictions.shape[-1])
    #     labels = labels.view(-1)
        
    #     # Define the loss function
    #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    #     # Compute the loss
    #     loss = loss_fct(predictions, labels)


    #     #loss = torch.tensor(trainer.eval_loss)
    #     perplexity = torch.exp(loss)
    #     metrics = {'perplexity': perplexity.item(), 'comp_loss': loss.item()}

    #     # Save metrics to a text file
    #     with open('metrics.txt', 'a') as file:
    #         file.write(f'Global step: {trainer.state.global_step}, Perplexity: {perplexity.item()}\n')

    #     print("Perplexity:", perplexity.item())

    #     return metrics

    # Use this to periodically trigger events during training
    class CustomCallback(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_epoch_end(self, args, state, control, **kwargs):
            # Perform custom evaluation at the end of each epoch
            if args.evaluation_strategy == "epoch":
                self.custom_evaluation()

        def on_log(self, args, state, control, **kwargs):
            global tic
            toc = time.time()
            print("Duration: ",(toc-tic)/60,"m")
            tic = toc

        def custom_evaluation(self):
            # Access the model
            print("\nEvaluation at the end of epoch:\n")
            results = self.trainer.evaluate()
            loss = results['eval_loss']
            perplexity = np.exp(loss)
            print("Loss:", loss)
            print("Perplexity:", perplexity)
            
            with open('metrics.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([results['epoch'], loss, perplexity])


    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
        # resume_from_checkpoint=True
        # resume_from_checkpoint=args.checkpoint_dir
    )

    trainer.add_callback(CustomCallback(trainer))

    # Train model
    trainer.train(resume_from_checkpoint=args.load_from_checkpoint) # More precise version would be to pass args.checkpoint_dir explicitly

    # Save model
    model.save_pretrained(args.output_dir)

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
