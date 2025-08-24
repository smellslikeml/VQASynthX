import json
import argparse
import re
from pathlib import Path

# Define keywords that indicate detail vs. vagueness
# Inspired by MECAT's goal of rewarding fine-grained descriptions (DATE metric)
SPATIAL_RELATION_KEYWORDS = [
    "left",
    "right",
    "behind",
    "front",
    "above",
    "below",
    "between",
    "next to",
    "on top of",
]
VAGUE_KEYWORDS = ["near", "far", "close", "around"]
METRIC_UNITS = ["meter", "meters", "m", "feet", "foot", "ft", "inch", "inches"]


def score_answer(prediction: str, ground_truth: str) -> dict:
    """
    Scores a single prediction based on its spatial detail.
    This is a conceptual adaptation of MECAT's DATE metric for the spatial domain.
    """
    prediction_lower = prediction.lower()
    score = 0
    detail_found = []

    # 1. Reward specific spatial relations (score: +1)
    relation_bonus = 0
    for keyword in SPATIAL_RELATION_KEYWORDS:
        if keyword in prediction_lower:
            relation_bonus = 1
            detail_found.append(f"relation: {keyword}")
            break
    score += relation_bonus

    # 2. Reward metric distance mentions (score: +2)
    metric_bonus = 0
    # Simple regex to find a number followed by a unit
    for unit in METRIC_UNITS:
        if re.search(r"\d+\s*" + re.escape(unit), prediction_lower):
            metric_bonus = 2  # Higher reward for metric detail
            detail_found.append(f"metric: {unit}")
            break
    score += metric_bonus

    # 3. Penalize vague terms if no other details are present (score: -1)
    vagueness_penalty = 0
    for keyword in VAGUE_KEYWORDS:
        if keyword in prediction_lower:
            if not detail_found:
                vagueness_penalty = -1
                detail_found.append(f"vague: {keyword}")
            break
    score += vagueness_penalty

    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "spatial_detail_score": score,
        "details": detail_found,
    }


def main(args):
    """
    Main function to read prediction/ground truth files and write scores.
    """
    predictions_path = Path(args.predictions_file)
    ground_truth_path = Path(args.ground_truth_file)
    output_path = Path(args.output_file)

    if not predictions_path.exists() or not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Input file not found. Checked: {predictions_path}, {ground_truth_path}"
        )

    with open(predictions_path, "r") as f_pred:
        predictions = [json.loads(line) for line in f_pred]
    with open(ground_truth_path, "r") as f_gt:
        ground_truths = [json.loads(line) for line in f_gt]

    # A real implementation would match by a common 'id' or 'question_id' key.
    # For this minimal example, we assume the lines are aligned.
    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Prediction and ground truth files have different number of lines."
        )

    results = []
    total_score = 0
    for pred_item, gt_item in zip(predictions, ground_truths):
        # Assuming items are dicts with an 'answer' key
        pred_answer = pred_item.get("answer", "")
        gt_answer = gt_item.get("answer", "")

        result = score_answer(pred_answer, gt_answer)
        # Carry over an identifier if it exists
        result["id"] = pred_item.get("id", gt_item.get("id"))
        results.append(result)
        total_score += result["spatial_detail_score"]

    avg_score = total_score / len(results) if results else 0

    output_data = {
        "average_spatial_detail_score": avg_score,
        "per_item_scores": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        json.dump(output_data, f_out, indent=2)

    print(f"Evaluation complete. Average Spatial Detail Score: {avg_score:.4f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VQA responses for spatial detail, inspired by MECAT's DATE metric."
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to the JSONL file with model predictions.",
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to the JSONL file with ground truth answers.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output/scores.json",
        help="Path to save the output scores JSON file.",
    )
    args = parser.parse_args()
    main(args)
