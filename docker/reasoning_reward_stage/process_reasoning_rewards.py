import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re

def_tokenizer = None
def_model = None


def get_prm_reward(prm_model, prm_tokenizer, questions, steps):
    """Calculates the reward for a given list of question-step pairs."""
    inputs = prm_tokenizer(
        questions,
        steps,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(prm_model.device)
    with torch.no_grad():
        logits = prm_model(**inputs).logits
        rewards = torch.squeeze(logits)
    return rewards.cpu().tolist()


def split_reasoning(text):
    """Splits the reasoning chain into steps based on newlines or sentence-ending punctuation."""
    # First, try splitting by newline, which is a common CoT step delimiter
    steps = [s.strip() for s in text.split("\n") if s.strip()]
    if len(steps) > 1:
        return steps

    # If newline splitting fails, fall back to sentence-based splitting
    # This regex splits by '.', '?', '!' followed by a space or end of string
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def process_batch(batch, args, prm_model, prm_tokenizer):
    """Processes a batch of data to add reasoning rewards."""
    questions_batch = []
    steps_batch = []
    indices_map = []

    for i, item in enumerate(batch):
        question = item.get(args.question_field, "")
        cot_text = item.get(args.cot_field, "")
        if not cot_text:
            item["reasoning_steps"] = []
            item["reasoning_step_scores"] = []
            continue

        steps = split_reasoning(cot_text)
        item["reasoning_steps"] = steps
        if not steps:
            item["reasoning_step_scores"] = []
            continue

        for step in steps:
            questions_batch.append(question)
            steps_batch.append(step)
        indices_map.append(len(steps))

    if not questions_batch:
        return batch

    rewards = get_prm_reward(prm_model, prm_tokenizer, questions_batch, steps_batch)

    # Handle case where only one reward is returned
    if not isinstance(rewards, list):
        rewards = [rewards]

    current_pos = 0
    for i, num_steps in enumerate(indices_map):
        if num_steps > 0:
            batch[i]["reasoning_step_scores"] = rewards[
                current_pos : current_pos + num_steps
            ]
            current_pos += num_steps
        else:
            batch[i]["reasoning_step_scores"] = []

    return batch


def main(args):
    print(f"Loading PRM model: {args.model_id_prm}")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.model_id_prm)
    prm_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id_prm, torch_dtype=torch.bfloat16
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prm_model.to(device)
    prm_model.eval()
    print(f"Model loaded on {device}")

    with open(args.input_path, "r") as f_in, open(args.output_path, "w") as f_out:
        lines = f_in.readlines()
        batch = []
        for line in tqdm(lines, desc="Processing data"):
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line: {line.strip()}")
                continue

            if len(batch) >= args.batch_size:
                processed_batch = process_batch(batch, args, prm_model, prm_tokenizer)
                for item in processed_batch:
                    f_out.write(json.dumps(item) + "\n")
                batch = []

        # Process the final batch
        if batch:
            processed_batch = process_batch(batch, args, prm_model, prm_tokenizer)
            for item in processed_batch:
                f_out.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score reasoning steps in model outputs using a Process Reward Model."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the output JSONL file.",
    )
    parser.add_argument(
        "--cot_field",
        type=str,
        required=True,
        help="The key in the JSON object that contains the CoT string.",
    )
    parser.add_argument(
        "--question_field",
        type=str,
        required=True,
        help="The key in the JSON object that contains the question.",
    )
    parser.add_argument(
        "--model_id_prm",
        type=str,
        default="UW-Madison-Lee-Lab/VersaPRM",
        help="The model ID for the Process Reward Model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing."
    )

    args = parser.parse_args()
    main(args)
