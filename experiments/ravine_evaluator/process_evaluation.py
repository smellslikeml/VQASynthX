import argparse
import json
import re
from datasets import load_dataset
import pandas as pd

def parse_answer(text):
    """Extracts the content from the <answer> tag."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()  # Fallback if no tag is found

def evaluate_predictions(predictions_df, ground_truth_ds):
    """
    Compares model predictions against ground truth answers.
    """
    # Pre-process ground truth to extract the answer
    def extract_gt_answer(example):
        # The ground truth answer is typically in the second turn of the conversation
        if len(example.get('conversations', [])) > 1 and example['conversations'][1].get('from') == 'gpt':
            return example['conversations'][1]['value']
        return None

    ground_truth_ds = ground_truth_ds.map(lambda x: {'ground_truth_answer': extract_gt_answer(x)})
    ground_truth_df = ground_truth_ds.to_pandas()

    # Ensure 'id' columns are of the same type for merging
    predictions_df['id'] = predictions_df['id'].astype(str)
    ground_truth_df['id'] = ground_truth_df['id'].astype(str)

    merged_df = pd.merge(predictions_df, ground_truth_df[['id', 'ground_truth_answer']], on='id')

    if merged_df.empty:
        print("Warning: No matching IDs found between predictions and ground truth. Check 'id' fields.")
        return {}

    # Parse the <answer> tag from the model's prediction
    merged_df['parsed_prediction'] = merged_df['prediction'].apply(parse_answer)

    # Basic exact match accuracy (case-insensitive)
    # A more sophisticated approach would handle numerical comparisons or fuzzy matching.
    correct_predictions = (merged_df['parsed_prediction'].str.lower() == merged_df['ground_truth_answer'].str.lower()).sum()
    total_predictions = len(merged_df)

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    metrics = {
        "total_evaluated": total_predictions,
        "correct_predictions": int(correct_predictions),
        "exact_match_accuracy": f"{accuracy:.2f}%"
    }

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions for VQA datasets inspired by RAVine.")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to the JSONL file with model predictions. Each line must have 'id' and 'prediction'.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the Hugging Face dataset to use as ground truth.")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use for evaluation (e.g., 'test', 'train').")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output metrics JSON file.")

    args = parser.parse_args()

    # Load model predictions
    try:
        predictions_df = pd.read_json(args.predictions_file, lines=True)
        if not all(col in predictions_df.columns for col in ['id', 'prediction']):
            raise ValueError("Predictions file must contain 'id' and 'prediction' columns.")
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        return

    # Load ground truth dataset
    try:
        ground_truth_ds = load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        return

    # Perform evaluation
    evaluation_metrics = evaluate_predictions(predictions_df, ground_truth_ds)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    print(f"Evaluation complete. Metrics saved to {args.output_file}")
    print(json.dumps(evaluation_metrics, indent=4))

if __name__ == "__main__":
    main()
