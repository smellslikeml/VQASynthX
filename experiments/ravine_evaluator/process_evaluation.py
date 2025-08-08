import argparse
import json
import re
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Conversion factors to meters
UNITS_TO_METERS = {
    'meters': 1.0,
    'meter': 1.0,
    'm': 1.0,
    'centimeters': 0.01,
    'cm': 0.01,
    'feet': 0.3048,
    'foot': 0.3048,
    'ft': 0.3048,
}

def parse_distance(text):
    """Parses distance and unit from a string, returns distance in meters."""
    # Regex to find a number followed by a unit
    match = re.search(r"(\d*\.?\d+)\s*([a-zA-Z]+)", text, re.IGNORECASE)
    if not match:
        return None

    value_str, unit_str = match.groups()
    value = float(value_str)
    unit_lower = unit_str.lower()

    # Find the matching unit in our dictionary
    for key, factor in UNITS_TO_METERS.items():
        if key in unit_lower:
            return value * factor
    return None

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with open(eval_file, 'r') as f:
        eval_data = [json.loads(line) for line in f]

    results = []
    errors = []

    for item in tqdm(eval_data, desc="Processing evaluation data"):
        image_path = item['image']
        question = item['question']
        ground_truth_answer = item.get('ground_truth_answer', '')

        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping item due to image loading error: {e}")
            continue

        # Qwen-VL chat format
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, [image], return_tensors='pt').to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        pred_dist_m = parse_distance(response_text)
        gt_dist_m = parse_distance(ground_truth_answer)

        error = None
        if pred_dist_m is not None and gt_dist_m is not None:
            error = abs(pred_dist_m - gt_dist_m)
            errors.append(error)
        
        results.append({
            'image': image_path,
            'question': question,
            'ground_truth': ground_truth_answer,
            'prediction': response_text,
            'gt_meters': gt_dist_m,
            'pred_meters': pred_dist_m,
            'absolute_error': error
        })

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"\nResults saved to {output_file}")

    if errors:
        mean_absolute_error = np.mean(errors)
        print(f"\n--- Evaluation Summary ---")
        print(f"Processed {len(errors)} items with valid distance pairs.")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f} meters")
    else:
        print("\nCould not compute metrics. No valid distance pairs found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a VLM on a spatial reasoning benchmark.")
    parser.add_argument('--model_id', type=str, required=True, help='Hugging Face model ID.')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to the .jsonl evaluation file.')
    parser.add_argument('--output_file', type=str, default='results.jsonl', help='Path to save the output results.')
    args = parser.parse_args()
    main(args)
