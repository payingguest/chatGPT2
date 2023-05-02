import torch
from transformers import DistilGPT2Tokenizer, DistilGPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Set up the tokenizer
tokenizer = DistilGPT2Tokenizer.from_pretrained('distilgpt2')

# Load your own language data
with open('path/to/your/data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the text
encodings = tokenizer(text)

# Convert the tokenized text to a PyTorch dataset
dataset = TextDataset(encodings)

# Set up the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the DistilGPT2 language model
model = DistilGPT2LMHeadModel.from_pretrained('distilgpt2')

# Fine-tune the model on your language data
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()
