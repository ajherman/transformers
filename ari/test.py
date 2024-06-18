# # from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# # device = "cuda"
# # model_id = "openai-community/gpt2"
# # model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
# # tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# # from datasets import load_dataset

# # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# # encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# # import torch
# # from tqdm import tqdm

# # max_length = model.config.n_positions
# # stride = 512
# # seq_len = encodings.input_ids.size(1)

# # nlls = []
# # prev_end_loc = 0
# # for begin_loc in tqdm(range(0, seq_len, stride)):
# #     end_loc = min(begin_loc + max_length, seq_len)
# #     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
# #     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
# #     target_ids = input_ids.clone()
# #     target_ids[:, :-trg_len] = -100

# #     with torch.no_grad():
# #         outputs = model(input_ids, labels=target_ids)
# #         neg_log_likelihood = outputs.loss

# #     nlls.append(neg_log_likelihood)

# #     prev_end_loc = end_loc
# #     if end_loc == seq_len:
# #         break

# # ppl = torch.exp(torch.stack(nlls).mean())
# # print("Perplexity: ",ppl)


# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
# from datasets import load_dataset
# import torch
# import numpy as np

# # Load the pretrained model and tokenizer
# model_name = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # Load the Wikitext-2 dataset
# eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1",split="validation")
# # eval_dataset = dataset['validation']

# # Preprocess the dataset
# def preprocess_function(examples):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# # Set up the training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_eval_batch_size=12,
# )

# # Define the compute_metrics function
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
#     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     perplexity = torch.exp(loss)
#     return {"perplexity": perplexity.item()}

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     eval_dataset=tokenized_eval_dataset,
#     compute_metrics=compute_metrics,
# )

# # Evaluate the model
# results = trainer.evaluate()
# print(f"Perplexity: {results['eval_perplexity']}")


from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np

# Load the pretrained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the Wikitext-2 dataset
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

# Preprocess the dataset in smaller chunks
max_length = model.config.n_positions
stride = 512

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=1,  # Use smaller batch size to manage memory
    dataloader_num_workers=4,
)

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)

# Evaluate the model in smaller chunks
chunk_size = 64  # Adjust chunk size as necessary
num_chunks = len(tokenized_eval_dataset) // chunk_size + 1
all_nlls = []

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(tokenized_eval_dataset))
    chunk = tokenized_eval_dataset.select(range(start_idx, end_idx))
    
    outputs = trainer.predict(chunk)
    all_nlls.append(outputs.metrics['eval_loss'])

# Calculate overall perplexity
avg_nll = np.mean(all_nlls)
perplexity = np.exp(avg_nll)
print(f"Perplexity: {perplexity}")
