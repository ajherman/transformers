from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the tokenizer and set the padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id
print(f"Padding token ID: {pad_token_id}")  # Should print a valid token ID

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Load and tokenize the dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(encode, batched=True)

# Define the compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    
    # Reshape predictions and labels to match expected shapes
    predictions = predictions.view(-1, predictions.shape[-1])
    labels = labels.view(-1)
    
    # Define the loss function with ignore_index set to the padding token ID
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # Compute the loss
    loss = loss_fct(predictions, labels)
    
    # Compute perplexity
    perplexity = torch.exp(loss)
    
    # Print for debugging
    print(f"Computed in metrics - Loss: {loss.item()}, Perplexity: {perplexity.item()}")
    
    return {'eval_loss': loss.item(), 'perplexity': perplexity.item()}

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=500,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)

# Perform evaluation
eval_results = trainer.evaluate()
print("Evaluation results:\n", eval_results)

# Manually compute perplexity using eval_loss
final_eval_loss = eval_results['eval_loss']
final_perplexity = torch.exp(torch.tensor(final_eval_loss))
print("Final Perplexity (Manual Calculation):", final_perplexity.item())
