import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

def main():
    parser = argparse.ArgumentParser(description="Run SFT inspired by Open-R1.")
    # Arguments from Open-R1 README
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config", type=str, default=None, help="The configuration name of the dataset to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--learning_rate", type=float, default=4.0e-5, help="The initial learning rate for AdamW.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--bf16", action='store_true', help="Whether to use bf16 (mixed) precision. Requires NVIDIA Ampere or higher.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the column in the dataset containing the text for SFT.")
    
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, name=args.dataset_config, split="train")

    print(f"Loading tokenizer for model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        report_to="none", # Disable wandb/tensorboard for this minimal example
    )

    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=dataset,
        dataset_text_field=args.text_column,
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    print("Starting training...")
    trainer.train()

    print(f"Training complete. Saving model to {args.output_dir}")
    trainer.save_model()

if __name__ == "__main__":
    main()
