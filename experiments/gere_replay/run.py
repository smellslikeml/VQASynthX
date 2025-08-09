import torch
import os
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, 
                            TrainingArguments, DataCollatorWithPadding)
from torch.utils.data import DataLoader, IterableDataset
import itertools

# This experiment script tests the core idea of GeRe: General Samples Replay.
# It simulates a continual learning scenario where we fine-tune a model on a 
# new, specific task (e.g., spatial VQA from VQASynth) while replaying 
# general-domain data to prevent catastrophic forgetting.

# --- 1. Define Datasets ---
# In a real scenario, this would be a VQASynth-generated dataset.
# We use a dummy dataset for this minimal, testable example.
spatial_vqa_data = {
    'text': [
        "Is the red chair to the left of the blue table?",
        "How many feet away is the monitor from the keyboard?",
        "The plant is approximately 3 meters from the window."
    ],
    'labels': [0, 1, 0]
}
spatial_vqa_dataset = Dataset.from_dict(spatial_vqa_data)

# This represents the 'general samples' for replay, as proposed by GeRe.
# We use the content from the SOURCE repo's yelp_train.json as a stand-in.
general_replay_data = {
    'text': [
        "This place is amazing! The food was delicious and the service was top-notch.",
        "A decent experience, nothing to write home about, but not bad either.",
        "I would not recommend this restaurant. The food was cold and the waiter was rude."
    ],
    'labels': [1, 0, 0] # Dummy labels for classification task
}
general_replay_dataset = Dataset.from_dict(general_replay_data)

# --- 2. Setup Model and Tokenizer ---
MODEL_NAME = 'distilbert-base-uncased'

# It's crucial to start from a pre-trained model whose general knowledge we want to preserve.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=False, truncation=True)

tokenized_spatial_dataset = spatial_vqa_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_general_dataset = general_replay_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# --- 3. Implement GeRe-inspired Replay Mechanism ---
# The core idea of GeRe is to mix general samples into the training process.
# We create a custom iterable dataset that yields batches mixing both datasets.
# A replay_ratio of 0.5 means batches are 50% spatial VQA, 50% general.

class MixedReplayDataset(IterableDataset):
    def __init__(self, specific_dataset, general_dataset, specific_batch_size, general_batch_size):
        self.specific_loader = DataLoader(specific_dataset, batch_size=specific_batch_size, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
        self.general_loader = DataLoader(general_dataset, batch_size=general_batch_size, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))

    def __iter__(self):
        # Cycle indefinitely through the smaller general dataset
        general_iterator = itertools.cycle(self.general_loader)
        for specific_batch in self.specific_loader:
            general_batch = next(general_iterator)
            
            # Combine batches
            combined_batch = {}
            for key in specific_batch.keys():
                combined_batch[key] = torch.cat([specific_batch[key], general_batch[key]], dim=0)
            
            yield combined_batch

    def __len__(self):
        return len(self.specific_loader)

# Instantiate the replay dataset
# Here we mix 2 samples from the specific task with 1 from the general task per batch
mixed_dataset = MixedReplayDataset(tokenized_spatial_dataset, tokenized_general_dataset, specific_batch_size=2, general_batch_size=1)

# --- 4. Configure and Run Trainer ---
# The standard Hugging Face Trainer is used. The replay logic is handled entirely
# by our custom dataset, demonstrating the principle's simplicity.

output_dir = "./vqasynth_gere_experiment_results"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1, # Batch size is controlled by the dataset loaders
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none", # Disable wandb/tensorboard for this minimal test
)

# The Trainer will now receive combined batches from our MixedReplayDataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mixed_dataset,
    # We don't need a data_collator here because our dataset yields pre-collated batches
)

if __name__ == "__main__":
    print("--- Starting GeRe-inspired replay training --- ")
    trainer.train()
    print("--- Training complete --- ")
    print(f"Model and logs saved to {output_dir}")
