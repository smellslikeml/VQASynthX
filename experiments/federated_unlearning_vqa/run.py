# Inspired by the federated unlearning pipeline in Oblivionis
# This script provides a minimal, self-contained experiment to demonstrate
# the concept of unlearning for a model fine-tuned on VQA-like data.

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import copy
import logging
import os

# Ensure reproducibility and set up environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_ID = "squad"  # Using SQuAD as a stand-in for a structured QA dataset
FORGET_RATIO = 0.1  # Percentage of data to "forget"
RANDOM_SEED = 42

# --- Helper Functions ---


def load_and_prepare_data():
    """Loads SQuAD, formats it, and splits into retain/forget sets."""
    logging.info(f"Loading and preparing dataset: {DATASET_ID}")
    # Use a small subset for faster execution
    dataset = (
        load_dataset(DATASET_ID, split="train")
        .shuffle(seed=RANDOM_SEED)
        .select(range(2000))
    )

    # Simple formatting for QA
    def format_prompt(example):
        if example["answers"]["text"]:
            return f"Question: {example['question']}\nAnswer: {example['answers']['text'][0]}"
        return None

    dataset = dataset.map(lambda x: {"text": format_prompt(x)})
    dataset = dataset.filter(lambda x: x["text"] is not None)

    # Split data
    split_index = int(len(dataset) * (1 - FORGET_RATIO))
    retain_data = dataset.select(range(split_index))
    forget_data = dataset.select(range(split_index, len(dataset)))
    full_data = dataset

    logging.info(
        f"Data split: {len(retain_data)} retain, {len(forget_data)} forget samples."
    )
    return retain_data, forget_data, full_data


def get_model_and_tokenizer():
    """Loads the model and tokenizer, applying LoRA config."""
    logging.info(f"Loading model and tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model, tokenizer


def train_model(model, tokenizer, train_dataset, output_dir):
    """Fine-tunes a model using the Hugging Face Trainer."""
    logging.info(f"Starting training for {output_dir}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        max_steps=100,  # Keep it short for a quick test
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["input_ids"] for f in data]),
        },
    )

    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True
    return model


def unlearn_model(model, tokenizer, forget_dataset, output_dir):
    """Performs unlearning using gradient ascent on the forget set, inspired by Oblivionis."""
    logging.info("Starting unlearning process (Gradient Ascent)")
    unlearning_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=-2e-4,  # Negative LR for gradient ASCENT
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_steps=20,  # Shorter than training
    )

    unlearner = Trainer(
        model=model,
        args=unlearning_args,
        train_dataset=forget_dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["input_ids"] for f in data]),
        },
    )

    model.config.use_cache = False
    unlearner.train()
    model.config.use_cache = True
    return model


@torch.no_grad()
def evaluate_model(model, dataset, tokenizer, description):
    """Evaluates the model's loss on a given dataset."""
    model.eval()
    # For a quick and consistent evaluation, use the Trainer's evaluate method
    eval_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=4,
        do_train=False,
        do_eval=True,
    )
    evaluator = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["input_ids"] for f in data]),
        },
    )
    metrics = evaluator.evaluate()
    avg_loss = metrics["eval_loss"]
    logging.info(f"Evaluation for '{description}': Average Loss = {avg_loss:.4f}")
    return avg_loss


def main():
    """Orchestrates the retain, finetune, unlearn, and evaluation pipeline."""
    base_model, tokenizer = get_model_and_tokenizer()

    retain_data, forget_data, full_data = load_and_prepare_data()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=256
        )

    retain_data = retain_data.map(
        tokenize_function, batched=True, remove_columns=retain_data.column_names
    )
    forget_data = forget_data.map(
        tokenize_function, batched=True, remove_columns=forget_data.column_names
    )
    full_data = full_data.map(
        tokenize_function, batched=True, remove_columns=full_data.column_names
    )

    logging.info("\n--- STEP 1: Training Retain Model (Gold Standard) ---")
    retain_model = train_model(
        copy.deepcopy(base_model), tokenizer, retain_data, "./outputs/retain_model"
    )

    logging.info("\n--- STEP 2: Training Full Model ---")
    full_model_trained = train_model(
        copy.deepcopy(base_model), tokenizer, full_data, "./outputs/full_model"
    )

    logging.info("\n--- STEP 3: Unlearn from Full Model ---")
    unlearned_model = unlearn_model(
        copy.deepcopy(full_model_trained),
        tokenizer,
        forget_data,
        "./outputs/unlearned_model",
    )

    logging.info("\n--- STEP 4: Final Evaluation ---")
    print("\n" + "=" * 50 + "\nEVALUATION RESULTS\n" + "=" * 50)

    print("\n--- On Retain Set ---")
    evaluate_model(retain_model, retain_data, tokenizer, "Retain Model")
    evaluate_model(full_model_trained, retain_data, tokenizer, "Full Model")
    evaluate_model(unlearned_model, retain_data, tokenizer, "Unlearned Model")

    print("\n--- On Forget Set ---")
    evaluate_model(retain_model, forget_data, tokenizer, "Retain Model (Baseline)")
    evaluate_model(
        full_model_trained, forget_data, tokenizer, "Full Model (Should be low loss)"
    )
    evaluate_model(
        unlearned_model, forget_data, tokenizer, "Unlearned Model (Should be high loss)"
    )

    print(
        "\n"
        + "=" * 50
        + "\nExperiment finished. Check the loss values above.\n"
        + "=" * 50
    )
    logging.info("SUCCESS CRITERIA:")
    logging.info(
        "1. Unlearned Model loss on Retain Set should be close to Retain Model loss."
    )
    logging.info(
        "2. Unlearned Model loss on Forget Set should be close to Retain Model loss (i.e., high)."
    )
    logging.info(
        "3. Full Model loss on Forget Set should be low, showing it learned the data."
    )


if __name__ == "__main__":
    main()
