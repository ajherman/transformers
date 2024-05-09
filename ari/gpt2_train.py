import torch
import sys

# Path to the 'src' directory of your local transformers repository
path_to_transformers = '../transformers/src'

# Prepend this path to sys.path
if path_to_transformers not in sys.path:
    sys.path.insert(0, path_to_transformers)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').map(tokenizer, batched=True)

# Tokenize the dataset
def encode(examples):
    return tokenizer(examples['text'])

dataset = dataset.map(encode, batched=True)


# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-wikitext", # output directory
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs=20, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    save_steps=10_000, # number of updates steps before checkpoint saves
    save_total_limit=2, # limit the total amount of checkpoints and deletes the older checkpoints
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./gpt2-wikitext")

