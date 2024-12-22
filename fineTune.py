from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Load the dataset
dataset = load_dataset('pubmed_qa', 'pqa_artificial', split='train[:1%]')  # Use 1% of the dataset for faster training

# Load the tokenizer and model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    # Tokenize the input text
    inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=512)
    # Set the labels to be the same as the inputs
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Reduce the number of epochs
    per_device_train_batch_size=1,  # Reduce the batch size
    save_steps=1000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()
