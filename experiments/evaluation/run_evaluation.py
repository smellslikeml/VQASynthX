import argparse
import json
import re
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on multiple-choice VQA datasets."
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to the ground truth JSONL file.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the model predictions JSONL file.",
    )
    return parser.parse_args()


def extract_answer(prediction_text, options_keys):
    """
    Extracts the final multiple-choice answer from a model's textual response.
    This logic is inspired by evaluation scripts for parsing LLM outputs on benchmarks
    like MedXpertQA.
    """
    # Case-insensitive search for the final answer delimiter
    parts = re.split(r"(?i)final answer", prediction_text)
    search_text = parts[-1]  # Focus on the text after "final answer" if it exists

    # Define a regex pattern based on the available option keys (e.g., A, B, C, D)
    option_pattern = f"[{''.join(options_keys)}]"

    # Try to find an uppercase letter match first
    letter_match = re.findall(option_pattern, search_text)
    if letter_match:
        # If "final answer" is present, the first match after it is likely the intended one.
        # Otherwise, the last match in the whole text is often the conclusion.
        return letter_match[0] if len(parts) > 1 else letter_match[-1]

    # If no uppercase match, try lowercase
    letter_match = re.findall(option_pattern.lower(), search_text)
    if letter_match:
        # Same logic as above, but convert to uppercase
        choice = letter_match[0] if len(parts) > 1 else letter_match[-1]
        return choice.upper()

    return None  # No valid answer found


def main():
    """Main function to run the evaluation."""
    args = parse_args()

    logging.info(f"Loading ground truth from: {args.ground_truth}")
    with open(args.ground_truth, "r") as f:
        ground_truth_data = {
            item["id"]: item for item in (json.loads(line) for line in f)
        }

    logging.info(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, "r") as f:
        predictions_data = [json.loads(line) for line in f]

    if len(ground_truth_data) != len(predictions_data):
        logging.warning(
            f"Mismatch in number of items. Ground truth: {len(ground_truth_data)}, "
            f"Predictions: {len(predictions_data)}. Matching by 'id'."
        )

    correct_count = 0
    total_count = 0
    unparsable_count = 0

    for pred in tqdm(predictions_data, desc="Evaluating predictions"):
        pred_id = pred.get("id")
        if not pred_id or pred_id not in ground_truth_data:
            logging.warning(
                f"Skipping prediction with missing or unmatched ID: {pred_id}"
            )
            continue

        gt = ground_truth_data[pred_id]
        gt_label = gt.get("label")

        # 'response' is a common key for model output text
        # 'content' is another possibility (e.g., from an API message)
        prediction_text = pred.get("response") or pred.get("content", "")

        # Get option keys (e.g., ['A', 'B', 'C']) from ground truth to build a robust parser
        options = gt.get("options", {})
        if not options:
            logging.warning(
                f"Skipping item {pred_id} due to missing 'options' in ground truth."
            )
            continue

        options_keys = list(options.keys())

        extracted_prediction = extract_answer(prediction_text, options_keys)

        if extracted_prediction is None:
            unparsable_count += 1
            logging.warning(
                f"Could not parse answer for ID {pred_id}. Response: '{prediction_text[:100]}...'"
            )
        elif extracted_prediction == gt_label:
            correct_count += 1

        total_count += 1

    if total_count == 0:
        logging.error("No matching predictions found to evaluate.")
        return

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\n------------------- Evaluation Results -------------------")
    print(f"Total Evaluated: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count - unparsable_count}")
    print(f"Unparsable: {unparsable_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    main()
