import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import re


# Basic tokenizer using regex and lowercasing.
# NLTK would be better but this avoids an extra dependency for a minimal example.
def simple_tokenizer(text):
    """A simple tokenizer that splits on non-alphanumeric characters and lowercases."""
    if not text:
        return []
    # Remove punctuation and split by space
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


def calculate_metrics(ground_truth, prediction):
    """
    Calculates precision, recall (completeness), and F1 score based on token overlap.
    This is a simplified proxy for RAVine's "nugget"-based evaluation, where tokens
    represent individual pieces of information or "facts".

    - Completeness (Recall): Measures how much of the ground truth is captured by the prediction.
    - Precision: Measures how much of the prediction is accurate according to the ground truth.
    - F1-Score: The harmonic mean of Precision and Completeness.
    """
    gt_tokens = set(simple_tokenizer(ground_truth))
    pred_tokens = set(simple_tokenizer(prediction))

    if not gt_tokens and not pred_tokens:
        return {"precision": 1.0, "completeness": 1.0, "f1": 1.0}
    if not pred_tokens:
        return {"precision": 0.0, "completeness": 0.0, "f1": 0.0}
    if not gt_tokens:
        return {"precision": 0.0, "completeness": 0.0, "f1": 0.0}

    intersection = gt_tokens.intersection(pred_tokens)

    precision = len(intersection) / len(pred_tokens)
    completeness = len(intersection) / len(
        gt_tokens
    )  # Recall is termed "Completeness" to align with RAVine

    if precision + completeness == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * completeness) / (precision + completeness)

    return {"precision": precision, "completeness": completeness, "f1": f1}


def run_evaluation(ground_truth_path, predictions_path):
    """
    Runs the evaluation by comparing model predictions to ground truth data.
    This script is inspired by the log-based evaluation in the RAVine project,
    which processes pre-generated model outputs against a set of ground truths.
    """
    logging.info(f"Loading ground truth from: {ground_truth_path}")
    with open(ground_truth_path, "r") as f:
        ground_truths = {item["id"]: item for item in (json.loads(line) for line in f)}

    logging.info(f"Loading predictions from: {predictions_path}")
    with open(predictions_path, "r") as f:
        predictions = {item["id"]: item for item in (json.loads(line) for line in f)}

    common_ids = set(ground_truths.keys()).intersection(set(predictions.keys()))
    if not common_ids:
        raise ValueError(
            "No common IDs found between ground truth and prediction files."
        )

    logging.info(f"Found {len(common_ids)} common items to evaluate.")

    total_metrics = defaultdict(float)

    for item_id in common_ids:
        gt_item = ground_truths[item_id]
        pred_item = predictions[item_id]

        gt_answer = gt_item.get("answer")
        pred_answer = pred_item.get("generated_answer")

        if gt_answer is None or pred_answer is None:
            logging.warning(
                f"Skipping item {item_id} due to missing 'answer' or 'generated_answer' field."
            )
            continue

        metrics = calculate_metrics(gt_answer, pred_answer)
        for key, value in metrics.items():
            total_metrics[key] += value

    # Calculate average metrics
    num_items = len(common_ids)
    final_results = {key: value / num_items for key, value in total_metrics.items()}
    final_results["evaluated_count"] = num_items
    final_results["total_ground_truth"] = len(ground_truths)
    final_results["total_predictions"] = len(predictions)

    # Task Completion Rate (inspired by RAVine's "Rate")
    completed_tasks = sum(
        1 for i in common_ids if predictions[i].get("generated_answer")
    )
    final_results["task_completion_rate"] = (
        completed_tasks / len(common_ids) if common_ids else 0.0
    )

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="""
    Evaluate VQA model outputs based on the RAVine framework's principles.
    This script compares generated answers against a ground truth file and
    calculates metrics like Completeness (Recall) and Precision.
    """
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to the ground truth JSONL file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the model predictions JSONL file.",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to save results as a JSON file."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        results = run_evaluation(args.ground_truth, args.predictions)

        results_json = json.dumps(results, indent=2)
        print(results_json)

        if args.output:
            with open(args.output, "w") as f:
                f.write(results_json)
            logging.info(f"Results saved to {args.output}")

    except (FileNotFoundError, ValueError) as e:
        logging.error(e)
        exit(1)


if __name__ == "__main__":
    main()
