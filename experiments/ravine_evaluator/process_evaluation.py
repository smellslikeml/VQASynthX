import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# A simple regex to parse the <think> and <answer> tags from model output
THINK_ANSWER_PATTERN = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)

def parse_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a VLM on a spatial reasoning dataset.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face model ID for the VLM to evaluate.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="Hugging Face dataset ID for the evaluation data.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="The split of the dataset to use for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results and metrics.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evaluate. If None, uses the entire dataset split.",
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_id):
    """Loads the VLM and its tokenizer with 4-bit quantization."""
    print(f"Loading model: {model_id}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer

def evaluate(args):
    """Main function to run the evaluation loop."""
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    
    print(f"Loading dataset: {args.dataset_id} (split: {args.dataset_split})")
    dataset = load_dataset(args.dataset_id, split=args.dataset_split)

    if args.sample_size:
        dataset = dataset.select(range(args.sample_size))
        print(f"Using a subsample of {args.sample_size} examples.")

    results = []
    correct_predictions = 0

    for example in tqdm(dataset, desc="Evaluating"):
        image = example["image"]
        question = example["question"]
        ground_truth = example["answer"]

        # Format the prompt for the model
        # This assumes a Qwen-VL like prompt format. May need adjustment for other models.
        query = tokenizer.from_list_format([
            {'image': image},
            {'text': f"Question: {question}\nAnswer with <think> and <answer> tags."},
        ])
        
        inputs = tokenizer(query, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Use greedy decoding for reproducibility
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the response to extract the final answer
        match = THINK_ANSWER_PATTERN.search(response_text)
        if match:
            predicted_answer = match.group(2).strip()
        else:
            # Fallback if the model doesn't follow the format
            predicted_answer = response_text.split("<answer>")[-1].strip()

        # Simple accuracy check (case-insensitive, ignores punctuation)
        # This is a basic metric. More sophisticated parsing would be needed for numerical answers.
        is_correct = predicted_answer.lower().strip('. ') == ground_truth.lower().strip('. ')
        
        if is_correct:
            correct_predictions += 1
            
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "full_response": response_text,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
        })

    # Calculate metrics
    total_samples = len(dataset)
    accuracy = (correct_predictions / total_samples) if total_samples > 0 else 0
    metrics = {"accuracy": accuracy, "total_samples": total_samples, "correct": correct_predictions}
    print(f"Evaluation finished. Accuracy: {accuracy:.4f}")

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "results.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
