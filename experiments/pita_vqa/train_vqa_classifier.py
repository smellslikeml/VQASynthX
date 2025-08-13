# Adapted from SOURCE: imdb_gen/train_classifier.py
# This script trains a PITA classifier on VQA preference data.

import argparse
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

from experiments.pita_vqa.classifier import CustomLlamaForClassification


def format_prompt(question, answer):
    # A simple prompt template for VQA
    return f"USER: <image>\n{question} ASSISTANT: {answer}"


class VQAPreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        chosen_answer = item["chosen"]
        rejected_answer = item["rejected"]

        chosen_text = format_prompt(question, chosen_answer)
        rejected_text = format_prompt(question, rejected_answer)

        chosen_tokens = self.tokenizer(
            chosen_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=256,
        )
        rejected_tokens = self.tokenizer(
            rejected_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=256,
        )

        return {
            "chosen_input_ids": chosen_tokens.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_tokens.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_tokens.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_tokens.attention_mask.squeeze(0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CustomLlamaForClassification(args.model_id).to(device)
    dataset = VQAPreferenceDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * args.num_epochs,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"Starting training on {device}...")
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            optimizer.zero_grad()

            chosen_logits = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            rejected_logits = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )

            # We are interested in the final token's logit for classification
            chosen_reward = chosen_logits[:, -1]
            rejected_reward = rejected_logits[:, -1]

            logits = chosen_reward - rejected_reward
            loss = loss_fn(logits, torch.ones_like(logits))

            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch+1} finished. Loss: {loss.item()}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(final_path, exist_ok=True)
    print(f"Saving final model checkpoint to {final_path}")
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))


if __name__ == "__main__":
    main()
