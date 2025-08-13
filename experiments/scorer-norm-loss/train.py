import argparse
import torch
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune LLaVA-NeXT with SCORER-inspired representation norm loss."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="The model ID to use for finetuning.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="remyxai/SpaceThinker",
        help="The VQA dataset to use for finetuning.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./llava-finetuned-scorer",
        help="Directory to save the finetuned model.",
    )
    # SCORER-inspired arguments
    parser.add_argument(
        "--use_norm_loss",
        action="store_true",
        help="Use representation norm loss on vision embeddings.",
    )
    parser.add_argument(
        "--coef_representation_norm",
        type=float,
        default=0.01,
        help="Coefficient for the representation norm loss.",
    )

    # Standard training arguments
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.4e-5,
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log every X updates steps."
    )

    return parser.parse_args()


class ScorerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.use_norm_loss = kwargs.pop("use_norm_loss", False)
        self.coef_representation_norm = kwargs.pop("coef_representation_norm", 0.0)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard cross-entropy loss from the base model
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # SCORER-inspired auxiliary loss
        if self.use_norm_loss and self.coef_representation_norm > 0:
            # Get the visual features (representations) from the vision tower.
            # This is analogous to the output of the PerceptionNetwork in SCORER.
            vision_tower_output = model.vision_tower(
                inputs["pixel_values"].to(model.dtype)
            )
            image_features = vision_tower_output[
                0
            ]  # The hidden states, before the pooler

            # Calculate the L2 norm of the image feature tensor
            representation_norm = torch.linalg.norm(image_features, ord=2)

            # Combine the standard LM loss with the auxiliary representation loss
            aux_loss = self.coef_representation_norm * representation_norm
            total_loss = lm_loss + aux_loss

            # Log the components for monitoring
            self.log({"lm_loss": lm_loss.item(), "rep_norm_loss": aux_loss.item()})
        else:
            total_loss = lm_loss

        return (total_loss, outputs) if return_outputs else total_loss


def main():
    args = parse_args()

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    processor = LlavaNextProcessor.from_pretrained(args.model_id)

    # Load a small subset of a VQASynth dataset for demonstration
    raw_dataset = load_dataset(args.dataset_name, split="train").select(range(100))

    def preprocess(example):
        # Extract the first user query and assistant response for a simple QA pair
        image = example["image"]
        prompt = "<image>\n" + example["conversations"][0]["value"]
        response = example["conversations"][1]["value"]
        full_text = prompt + " " + response + processor.tokenizer.eos_token

        inputs = processor(
            text=full_text,
            images=image,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs["labels"] = inputs.input_ids.clone()

        # Mask the prompt part so loss is only calculated on the response
        prompt_tokens = processor(text=prompt, return_tensors="pt").input_ids.shape[1]
        inputs["labels"][:, :prompt_tokens] = -100

        # Remove the batch dimension added by the processor
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs

    processed_dataset = raw_dataset.map(
        preprocess, remove_columns=raw_dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        report_to="none",
        bf16=True,
    )

    trainer = ScorerTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        use_norm_loss=args.use_norm_loss,
        coef_representation_norm=args.coef_representation_norm,
    )

    print(
        f"Starting training with use_norm_loss={args.use_norm_loss} and coef={args.coef_representation_norm}"
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model and processor saved to {args.output_dir}")


if __name__ == "__main__":
    main()
