import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a VQASynth-trained model on a spatial reasoning dataset.")
    parser.add_argument("--model_id", type=str, default="remyxai/SpaceThinker-Qwen2.5VL-3B", help="Hugging Face model ID.")
    parser.add_argument("--dataset_id", type=str, default="remyxai/OpenSpaces_MC_R1", help="Hugging Face dataset ID.")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use for evaluation.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate. -1 for all.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the results CSV.")
    return parser.parse_args()

def extract_numeric_answer(text):
    """Extract the first numeric value from a string."""
    if text is None:
        return None
    # Matches integers and floats, including those with units like 'cm' or 'meters'
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
    return float(matches[0]) if matches else None

def main():
    args = parse_args()
    print(f"Starting evaluation with model: {args.model_id} on dataset: {args.dataset_id}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load model and processor
    print("Loading model and processor...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map=device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    print("Model and processor loaded.")

    # Load dataset
    print(f"Loading dataset '{args.dataset_id}' split '{args.dataset_split}'...")
    try:
        dataset = load_dataset(args.dataset_id, split=args.dataset_split)
    except Exception as e:
        print(f"Could not load split '{args.dataset_split}'. Attempting to load 'train'. Error: {e}")
        dataset = load_dataset(args.dataset_id, split="train")

    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples.")

    results = []
    qualitative_correct = 0
    total_qualitative = 0
    total_absolute_error = 0.0
    total_quantitative = 0

    for item in tqdm(dataset, desc="Evaluating samples"):
        image = item["image"]
        
        # This assumes a standard 'conversations' format. May need adjustment.
        # We will take the first user turn as the question.
        question = ""
        if 'conversations' in item and len(item['conversations']) > 0:
            question = item['conversations'][0]['value']
        elif 'question' in item:
            question = item['question']
        else:
            # Fallback for unexpected format
            continue 

        ground_truth_answer = ""
        if 'conversations' in item and len(item['conversations']) > 1:
            ground_truth_answer = item['conversations'][1]['value']
        elif 'answer' in item:
            ground_truth_answer = item['answer']
        else:
            continue

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{question}"}
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device, torch_dtype)

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        model_answer = generated_texts[0].split("<|assistant|>")[1].strip()

        # Evaluate the answer
        gt_numeric = extract_numeric_answer(ground_truth_answer)
        model_numeric = extract_numeric_answer(model_answer)
        
        row = {
            "question": question,
            "ground_truth": ground_truth_answer,
            "model_answer": model_answer,
            "gt_numeric": gt_numeric,
            "model_numeric": model_numeric,
            "absolute_error": None
        }

        if gt_numeric is not None and model_numeric is not None:
            # Quantitative evaluation
            absolute_error = abs(gt_numeric - model_numeric)
            row["absolute_error"] = absolute_error
            total_absolute_error += absolute_error
            total_quantitative += 1
        else:
            # Qualitative evaluation (simple case-insensitive match for keywords)
            gt_lower = ground_truth_answer.lower()
            model_lower = model_answer.lower()
            # Simple check for common affirmative/negative keywords
            positive_keywords = ["yes", "correct", "indeed", "greater"]
            negative_keywords = ["no", "not", "incorrect", "less"]
            is_gt_pos = any(kw in gt_lower for kw in positive_keywords)
            is_model_pos = any(kw in model_lower for kw in positive_keywords)
            is_gt_neg = any(kw in gt_lower for kw in negative_keywords)
            is_model_neg = any(kw in model_lower for kw in negative_keywords)

            if (is_gt_pos and is_model_pos) or (is_gt_neg and is_model_neg):
                qualitative_correct += 1
            
            total_qualitative += 1

        results.append(row)

    # Final Metrics
    print("\n--- Evaluation Summary ---")
    if total_quantitative > 0:
        mean_absolute_error = total_absolute_error / total_quantitative
        print(f"Quantitative Questions: {total_quantitative}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
    else:
        print("No quantitative questions found.")

    if total_qualitative > 0:
        qualitative_accuracy = (qualitative_correct / total_qualitative) * 100
        print(f"Qualitative Questions: {total_qualitative}")
        print(f"Qualitative Accuracy: {qualitative_accuracy:.2f}%")
    else:
        print("No qualitative questions found.")
    
    # Save results to CSV
    output_path = Path(args.output_dir) / "evaluation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"--- Results saved to {output_path} ---")


if __name__ == "__main__":
    main()
