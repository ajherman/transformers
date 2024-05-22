import torch
import sys
import argparse
import csv
import os

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
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import logging

# Read in config file
config = GPT2Config.from_json_file(args.config_file)

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Add pad token

# Create model from config
model = AutoModelForCausalLM.from_config(config)

print("Config:\n", model.config)

# Print number of model parameters
num_params = sum(p.numel() for p in model.parameters())
print("Number of model parameters:", num_params)

# Load the dataset
logging.getLogger("datasets").setLevel(logging.DEBUG)

train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
print("Finished loading datasets")

# Tokenize the dataset
def encode(examples):
    return tokenizer(examples['text'], truncation=True, max_length=model.config.n_ctx, padding='max_length')

train_dataset = train_dataset.map(encode, batched=True)
eval_dataset = eval_dataset.map(encode, batched=True)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,  # output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=args.num_train_epochs,  # number of training epochs
    per_device_train_batch_size=args.per_device_train_batch_size,  # batch size for training
    save_steps=args.save_steps,  # number of updates steps before checkpoint saves
    save_total_limit=args.save_total_limit,  # limit the total amount of checkpoints and deletes the older checkpoints
    evaluation_strategy="steps",  # evaluation strategy to adopt during training
    eval_steps=1000,  # number of steps before evaluation
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Ensure predictions and labels are tensors
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    
    # Reshape predictions and labels to be compatible with CrossEntropyLoss
    predictions = predictions.view(-1, predictions.shape[-1])
    labels = labels.view(-1)
    
    # Define the loss function with ignore_index set to the padding token ID
    loss_fct = torch.nn.CrossEntropyLoss()
    
    # Compute the loss
    loss = loss_fct(predictions, labels)
    
    # Compute perplexity
    perplexity = torch.exp(loss)
    
    # Print for debugging
    print(f"Computed in metrics - Loss: {loss.item()}, Perplexity: {perplexity.item()}")
    
    return {'eval_loss': loss.item(), 'perplexity': perplexity.item()}

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train(resume_from_checkpoint=False)

# Save model
model.save_pretrained(args.output_dir)

# Evaluate model
eval_results = trainer.evaluate()
print("Evaluation results:\n", eval_results)

# Manually compute perplexity using eval_loss
final_eval_loss = eval_results['eval_loss']
final_perplexity = torch.exp(torch.tensor(final_eval_loss))
print("Final Perplexity (Manual Calculation):", final_perplexity.item())
