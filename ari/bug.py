import torch
import os 

os.environ["HF_DATASETS_OFFLINE"] = "0"

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, AutoModelForCausalLM, GPT2Config
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerControl
from datasets import load_dataset



config = GPT2Config()

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Add pad token
model = AutoModelForCausalLM.from_config(config)

dataset_name = "monology/pile-uncopyrighted"
train_dataset = load_dataset(dataset_name, split='train', streaming=True, block_size=655360)

# Tokenize the dataset
def encode(examples):

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
    output_dir='tmp', # output directory
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train model
trainer.train() # More precise version would be to pass args.checkpoint_dir explicitly

