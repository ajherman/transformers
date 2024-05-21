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
tokenizer.pad_token = tokenizer.eos_token # Add pad token
# model = GPT2LMHeadModel.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(config)

print("Config:\n", model.config)

# Print number of model parameters
num_params = sum(p.numel() for p in model.parameters())
print("Number of model parameters:", num_params)

# Load the dataset
logging.getLogger("datasets").setLevel(logging.DEBUG)

train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').map(tokenizer, batched=True)
print("Finished loading datasets")
#assert(0)
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
    evaluation_strategy="steps", # evaluation strategy to adopt during training
    eval_steps=1000, # number of steps before evaluation
    # warmup_steps=500,                # number of warmup steps for learning rate scheduler
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions, labels = torch.from_numpy(predictions), torch.from_numpy(labels)
    predictions = predictions.view(-1,predictions.shape[-1])
    labels = labels.view(-1)
    loss = torch.nn.functional.cross_entropy(predictions, labels)
   
    #loss = torch.tensor(trainer.eval_loss)
    perplexity = torch.exp(loss)
    metrics = {'perplexity': perplexity.item(), 'comp_loss': loss.item(), 'ppl':np.exp(loss.item())}

    # Save metrics to a text file
    with open('metrics.txt', 'a') as file:
        file.write(f'Global step: {trainer.state.global_step}, Perplexity: {perplexity.item()}\n')

    print("Perplexity:", perplexity.item())

    return metrics

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    # resume_from_checkpoint=True
    # resume_from_checkpoint=args.checkpoint_dir
)
# Train model
trainer.train(resume_from_checkpoint=False) # More precise version would be to pass args.checkpoint_dir explicitly

# Save model
model.save_pretrained(args.output_dir)

# Evaluate model
eval_results = trainer.evaluate()
print("Evaluation results:\n", eval_results)

# Perplexity
perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
print("Perplexity:", perplexity)



