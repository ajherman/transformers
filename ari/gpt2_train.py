
import torch
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='GPT-2 Training')
parser.add_argument('--output_dir', type=str, default='./gpt2-wikitext', help='Output directory')
parser.add_argument('--num_train_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--save_steps', type=int, default=10_000, help='Number of updates steps before checkpoint saves')
parser.add_argument('--save_total_limit', type=int, default=2, help='Limit the total amount of checkpoints and deletes the older checkpoints')
parser.add_argument('--use_local_transformers', action='store_true', help='Use local transformers repository')
args = parser.parse_args()

# Path to the 'src' directory of your local transformers repository
use_local_transformers = args.use_local_transformers
if use_local_transformers:
    # Path to the 'src' directory of your local transformers repository
    path_to_transformers = '../transformers/src'

    # Prepend this path to sys.path
    if path_to_transformers not in sys.path:
        sys.path.insert(0, path_to_transformers)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Add pad token
# model = GPT2LMHeadModel.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
config = model.config

print("Config:\n", config)

# Load the dataset
train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').map(tokenizer, batched=True)

# Tokenize the dataset
def encode(examples):
    return tokenizer(examples['text'])

train_dataset = train_dataset.map(encode, batched=True)
eval_dataset = eval_dataset.map(encode, batched=True)


# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir, # output directory
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs=args.num_train_epochs, # number of training epochs
    per_device_train_batch_size=args.per_device_train_batch_size, # batch size for training
    save_steps=args.save_steps, # number of updates steps before checkpoint saves
    save_total_limit=args.save_total_limit, # limit the total amount of checkpoints and deletes the older checkpoints
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained(args.output_dir)

# Evaluate model
eval_results = trainer.evaluate()
print("Evaluation results:\n", eval_results)

# Perplexity
perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
print("Perplexity:", perplexity)



