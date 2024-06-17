# from transformers import TrainingArguments, Trainer, GPT2Tokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Config, AutoTokenizer
# from datasets import load_dataset, load_metric

# # training_args = TrainingArguments("test_trainer"),

# import numpy as np
# import torch
# import datasets
# import evaluate

# # # Load pre-trained model and tokenizer
# # model_name = 'gpt2'
# # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# # tokenizer.pad_token = tokenizer.eos_token # Add pad token
# # model = AutoModelForCausalLM.from_pretrained(model_name)

# # dataset_name = "monology/pile-uncopyrighted"
# # # train_dataset = load_dataset(dataset_name, split='train', streaming=True, block_size=655360)
# # eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
# # # train_encodings = tokenizer(train_dataset, return_tensors="pt", padding=True, truncation=True)
# # eval_encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")


# # # metric = load_metric("accuracy")
# # metric = load_metric("perplexity")

# # def compute_metrics(eval_pred):
# #     logits, labels = eval_pred
# #     predictions = np.argmax(logits, axis=-1)
# #     return metric.compute(predictions=predictions, references=labels)

# # trainer = Trainer(
# #     model=model,
# #     # args=training_args,
# #     # train_dataset=small_train_dataset,
# #     eval_dataset=eval_encodings,
# #     compute_metrics=compute_metrics,
# # )

# # # Evaluate model
# # eval_results = trainer.evaluate()
# # print("Evaluation results:\n", eval_results)

# # # Perplexity
# # perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
# # print("Perplexity:", perplexity)



# perplexity = evaluate.load("perplexity", module_type="metric")
# input_texts = datasets.load_dataset("wikitext",
#                                     "wikitext-2-raw-v1",
#                                     split="test")["text"][:50]
# input_texts = [s for s in input_texts if s!='']
# results = perplexity.compute(model_id='gpt2',
#                              predictions=input_texts)
# print(list(results.keys()))
# print(round(results["mean_perplexity"], 2))
# print(round(results["perplexities"][0], 2))

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "openai-community/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
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

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())