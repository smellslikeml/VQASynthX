import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import load_dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# A key constant for SFT loss calculation
IGNORE_INDEX = -100


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Collate examples for supervised fine-tuning.
    Inspired by LLaMA-Factory's data collation.
    """

    processor: AutoProcessor

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        # Pad sequences to the max length in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbl) for lbl in labels],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        # Create attention mask
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        # Handle pixel values for images
        if "pixel_values" in instances[0]:
            pixel_values = [instance["pixel_values"] for instance in instances]
            batch["pixel_values"] = torch.stack(pixel_values)

        return batch


def main():
    parser = argparse.ArgumentParser(
        description="LLaVA QLoRA SFT Script inspired by LLaMA-Factory"
    )
    # Model and Data Arguments from LLaMA-Factory's hparams
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the JSONL dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the LoRA adapter",
    )

    # Training Arguments inspired by transformers.TrainingArguments and LLaMA-Factory
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for AdamW"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 for training (recommended for Ampere+)",
    )

    # LoRA Arguments from LLaMA-Factory's finetuning_args
    parser.add_argument(
        "--lora_r", type=int, default=64, help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=128, help="LoRA alpha scaling"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        help="Modules to apply LoRA to",
    )

    args = parser.parse_args()

    # --- 1. Load Model and Processor ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- 2. Prepare model for QLoRA ---
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 3. Load and Preprocess Dataset ---
    def preprocess_function(example):
        image = Image.open(example["image"]).convert("RGB")
        prompt = processor.tokenizer.apply_chat_template(
            example["conversations"], tokenize=False, add_generation_prompt=False
        )
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        labels = inputs.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX
        return {
            "input_ids": inputs.input_ids[0].tolist(),
            "labels": labels[0].tolist(),
            "pixel_values": inputs.pixel_values[0],
        }

    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    dataset = raw_dataset.map(
        preprocess_function, remove_columns=next(iter(raw_dataset)).keys()
    )

    # --- 4. Configure Trainer ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=args.bf16,
        fp16=not args.bf16,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSupervisedDataset(processor=processor),
    )

    # --- 5. Start Training ---
    print("Starting QLoRA SFT training...")
    trainer.train()

    # --- 6. Save Final Adapter ---
    final_save_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"Final LoRA adapter and processor saved to {final_save_path}")


if __name__ == "__main__":
    main()
