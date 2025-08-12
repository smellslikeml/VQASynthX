import argparse
import json
import re
from collections import Counter

def extract_nuggets(text: str) -> set:
    """
    A simple nugget extractor based on keywords and numbers.
    This is a placeholder for a more sophisticated, possibly LLM-based, extractor.
    It finds numbers, units, and keywords related to spatial relationships.
    Example: "The chair is about 2.5 meters to the left of the table."
    -> {"2.5", "meters", "left of"}
    """
    # Find all numbers (integers or floats)
    numbers = set(re.findall(r'\d+\.?\d*', text))

    # Find common units of measurement
    units = set(re.findall(r'\b(meters|meter|feet|foot|inches|inch|cm|m)\b', text, re.IGNORECASE))

    # Find common spatial prepositions/relations
    relations = set(re.findall(r'\b(left of|right of|in front of|behind|above|below|near|far from|close to)\b', text, re.IGNORECASE))
    
    # Normalize to lowercase
    nuggets = numbers.union({u.lower() for u in units}).union({r.lower() for r in relations})
    
    return nuggets

def calculate_metrics(ground_truth_nuggets: set, predicted_nuggets: set):
    """Calculates precision, recall, and F1 score for the predicted nuggets."""
    if not ground_truth_nuggets and not predicted_nuggets:
        return 1.0, 1.0, 1.0
    if not predicted_nuggets:
        # No prediction, precision is perfect (debatable, but avoids division by zero) but recall is 0
        return 1.0, 0.0, 0.0
    if not ground_truth_nuggets:
        # Prediction exists but no ground truth, recall is perfect but precision is 0
        return 0.0, 1.0, 0.0

    true_positives = len(ground_truth_nuggets.intersection(predicted_nuggets))
    
    precision = true_positives / len(predicted_nuggets)
    recall = true_positives / len(ground_truth_nuggets)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
            
    return precision, recall, f1_score

def main(args):
    ground_truths = {}
    with open(args.ground_truth, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Assuming a structure like {"id": "...", "conversations": [{"value": "..."}, {"value": "..."}]}
            # where the second conversation value is the answer.
            if 'id' in data and 'conversations' in data and len(data['conversations']) > 1:
                ground_truths[data['id']] = data['conversations'][1]['value']

    predictions = {}
    with open(args.predictions, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Assuming a structure like {"id": "...", "prediction": "..."}
            if 'id' in data and 'prediction' in data:
                predictions[data['id']] = data['prediction']
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0
    
    print(f"{'ID':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)

    for qid, gt_answer in ground_truths.items():
        if qid in predictions:
            pred_answer = predictions[qid]
            
            gt_nuggets = extract_nuggets(gt_answer)
            pred_nuggets = extract_nuggets(pred_answer)
            
            if not gt_nuggets:
                continue # Skip items with no extractable ground truth nuggets

            precision, recall, f1 = calculate_metrics(gt_nuggets, pred_nuggets)
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1
            
            print(f"{qid:<15} | {precision:<10.2f} | {recall:<10.2f} | {f1:<10.2f}")

    if count > 0:
        avg_precision = total_precision / count
        avg_recall = total_recall / count
        avg_f1 = total_f1 / count
        
        print("-" * 55)
        print("Average Metrics:")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f}")
        print(f"  F1-Score:  {avg_f1:.4f}")
    else:
        print("No matching items found to evaluate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions using a nugget-based approach inspired by RAVine.")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to the ground truth JSONL file from VQASynth.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the model predictions JSONL file.")
    args = parser.parse_args()
    main(args)
