from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer
import numpy as np
import os
device = "cuda"
model_id = "openai-community/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

local_rank = int(os.getenv('LOCAL_RANK', '0'))


from datasets import load_dataset
import torch
from tqdm import tqdm
import torch.distributed as dist

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",shuffle=True)
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = model.config.n_positions  #  512
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
print("Perplexity: ",ppl)

######################################################################

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Preprocess the dataset in smaller chunks
max_length = model.config.n_positions

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

tokenized_eval_dataset = test.map(preprocess_function, batched=True)
tokenized_eval_dataset.set_format(type='torch',columns=['input_ids','attention_mask'])

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16,  # Use smaller batch size to manage memory
    dataloader_num_workers=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_eval_dataset,
)
all_losses = []

# Manually compute the loss for each chunk
for batch in trainer.get_eval_dataloader():
    input_ids = batch["input_ids"].to("cuda")
    #print(input_ids)
    #assert(0)
    
    labels = input_ids.clone()
    attention_mask = batch["attention_mask"].to("cuda")
    labels[attention_mask==0]=-100
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        print(attention_mask.sum().item())
    if attention_mask.sum().item()>=1200:
        all_losses.append(loss.item())

# if dist.is_initialized():
#     dist.all_reduce(torch.tensor(all_losses), op=dist.ReduceOp.SUM)
#     all_losses = all_losses.tolist()

if local_rank == 0:

    # Calculate overall perplexity
    print(all_losses)
    avg_loss = np.mean(all_losses)
    perplexity = np.exp(avg_loss)
    print(f"Perplexity: {perplexity}")
