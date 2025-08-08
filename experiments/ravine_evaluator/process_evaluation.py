import argparse
import json
import re
import numpy as np
from typing import Dict, List, Optional

def convert_to_meters(value: float, unit: str) -> float:
    """Converts a distance to meters."""
    unit = unit.lower()
    if unit in ["meter", "meters", "m"]:
        return value
    elif unit in ["centimeter", "centimeters", "cm"]:
        return value / 100.0
    elif unit in ["foot", "feet", "ft"]:
        return value * 0.3048
    # Assume meters if unit is unknown or not specified
    return value

def parse_distance(text: str) -> Optional[float]:
    """
    Parses a string to find a numerical distance and converts it to meters.
    Example: "The distance is 2.5 feet." -> 0.762
    """
    # Regex to find a number followed by an optional common distance unit.
    match = re.search(r'(\d+\.?\d*)\s*(meters?|m|centimeters?|cm|feet|foot|ft)\b', text, re.IGNORECASE)
    
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return convert_to_meters(value, unit)
        
    # Fallback for just a number, assuming the canonical unit is meters.
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA model predictions based on RAVine principles.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the JSONL file with model predictions.")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to the JSONL file with ground truth answers.")
    parser.add_argument("--id-key", type=str, default="image_id", help="The key used to match predictions and ground truth.")
    
    args = parser.parse_args()

    # Load ground truth into a dictionary for easy lookup
    ground_truth_map: Dict[str, Dict] = {}
    with open(args.ground_truth, 'r') as f:
        for line in f:
            data = json.loads(line)
            if args.id_key in data:
                ground_truth_map[data[args.id_key]] = data

    predictions: List[Dict] = []
    with open(args.predictions, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))

    errors = []
    parsed_count = 0
    total_count = 0

    for pred in predictions:
        total_count += 1
        pred_id = pred.get(args.id_key)
        if not pred_id or pred_id not in ground_truth_map:
            continue

        gt = ground_truth_map[pred_id]
        
        # Extract text from the <answer> tag if present, otherwise use the whole string
        pred_answer = pred.get('answer', '')
        if '<answer>' in pred_answer:
            pred_answer = pred_answer.split('<answer>')[-1].strip()

        gt_answer = gt.get('answer', '')

        pred_dist = parse_distance(pred_answer)
        gt_dist = parse_distance(gt_answer)

        if pred_dist is not None and gt_dist is not None:
            errors.append(abs(pred_dist - gt_dist))
            parsed_count += 1

    print("--- VQASynth Evaluation Report (inspired by RAVine) ---")
    if total_count > 0:
        print(f"Processed {total_count} predictions.")
        print(f"Successfully parsed quantitative distances for {parsed_count} pairs.")
    
    if errors:
        mae = np.mean(errors)
        std_err = np.std(errors)
        median_ae = np.median(errors)
        
        print("\nQuantitative Distance Estimation Metrics (in meters):")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Median Absolute Error:     {median_ae:.4f}")
        print(f"  Standard Deviation of AE:  {std_err:.4f}")
    else:
        print("\nCould not compute metrics. No valid quantitative distance pairs found.")
        print("Please ensure your prediction and ground truth files contain parseable distances (e.g., '5 meters', '1.2 ft').")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    main()
