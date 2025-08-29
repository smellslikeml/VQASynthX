import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import logging

# Use the mock trainer that simulates the GeRe API
from gere_mock.trainer import GeReTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # --- 1. Configuration ---
    # For this minimal example, we use a small, publicly available model.
    # The GeRe demo uses a custom tiny LLaMA, but distilbert is easier for a self-contained example.
    model_name = "distilbert-base-uncased"
    output_dir = "./results"
    gere_cache_dir = "./gere_cache"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5,
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if available
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",  # Disable wandb/tensorboard for this minimal example
    )

    # --- 2. Load Model and Tokenizer ---
    logger.info(f"Loading model and tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # The VQASynth task is ultimately a classification or generation task.
    # We'll use SequenceClassification as a proxy for the finetuning task.
    # The dummy dataset has 3 unique labels: 'near', 'far', 'medium'.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # --- 3. Load "New Task" Dataset (Spatial VQA) ---
    logger.info("Loading dummy spatial VQA dataset...")
    dataset = load_dataset("json", data_files="dummy_spatial_vqa.jsonl", split="train")

    # --- 4. Preprocess Dataset ---
    labels_list = sorted(list(set(dataset["answer"])))
    text_label_to_int = {label: i for i, label in enumerate(labels_list)}

    def preprocess_function(examples):
        sentences = [
            f"Question: {q} Answer: {a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        tokenized_inputs = tokenizer(sentences, truncation=True, padding=False)
        tokenized_inputs["labels"] = [text_label_to_int[a] for a in examples["answer"]]
        return tokenized_inputs

    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 5. Initialize GeRe Trainer ---
    logger.info("Initializing GeReTrainer...")
    trainer = GeReTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # GeRe-specific configurations from the SOURCE repo README
        gere_hidden_state_saving_dir=gere_cache_dir,
        reuse_gere_hidden_state=True,
        num_interpolate_per_batch=0,  # Corresponds to the MMLU results example
        w_strategy="dy",  # Corresponds to the MMLU results example
    )

    # --- 6. Train ---
    logger.info("Starting training with GeReTrainer...")
    trainer.train()
    logger.info("Training complete.")

    # --- 7. Save Artifacts ---
    logger.info(f"Saving final model and tokenizer to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
