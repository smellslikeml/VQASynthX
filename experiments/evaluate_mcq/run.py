import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm


def parse_final_answer(response_text: str) -> str | None:
    """
    Parses the model's text response to extract the final multiple-choice answer.
    This logic is adapted from the answer parsing in the GPT-5-Evaluation repository,
    specifically from `eval_medxpertqa.py`.

    It looks for a 'final answer' sentinel and then extracts the first capital letter
    (A-J) following it. If not found, it defaults to the last capital letter
    in the entire response.

    Args:
        response_text: The full text output from the language model.

    Returns:
        The extracted single-letter answer (e.g., "A") or None if no answer is found.
    """
    # Split the response by "final answer" case-insensitively
    parts = re.split(r"(?i)final answer", response_text)

    # The part after "final answer" is the last element
    final_answer_part = parts[-1]

    # Look for patterns like (A), A., A)
    pattern = r"\(?([A-J])\)?"
    match = re.search(pattern, final_answer_part)

    if match:
        return match.group(1).upper()

    # Fallback for less structured answers, find the last mentioned letter
    all_letters = re.findall(r"[A-J]", response_text)
    if all_letters:
        return all_letters[-1].upper()

    return None


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM's multiple-choice question answering performance."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the .jsonl file containing model predictions. Each line must be a JSON object with 'id' and 'response' keys.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to the .jsonl file containing ground truth data. Each line must be a JSON object with 'id' and 'label' keys.",
    )
    args = parser.parse_args()

    # Load ground truth data into a dictionary for easy lookup
    ground_truth = {}
    with open(args.ground_truth, "r") as f:
        for line in f:
            data = json.loads(line)
            ground_truth[data["id"]] = data["label"]

    print(f"Loaded {len(ground_truth)} ground truth examples.")

    # Load predictions and evaluate
    predictions = []
    with open(args.predictions, "r") as f:
        for line in f:
            predictions.append(json.loads(line))

    print(f"Loaded {len(predictions)} predictions.")

    if not predictions:
        print("No predictions found. Exiting.")
        return

    correct_count = 0
    total_count = 0

    for pred in tqdm(predictions, desc="Evaluating"):
        pred_id = pred.get("id")
        if pred_id is None or pred_id not in ground_truth:
            print(f"Warning: Skipping prediction with missing or unmatched ID: {pred}")
            continue

        true_label = ground_truth[pred_id]
        response_text = pred.get("response")

        if response_text is None:
            print(
                f"Warning: Skipping prediction with missing 'response' field for ID {pred_id}"
            )
            continue

        parsed_answer = parse_final_answer(response_text)

        if parsed_answer and parsed_answer == true_label:
            correct_count += 1

        total_count += 1

    if total_count == 0:
        print("No matching predictions found to evaluate.")
        accuracy = 0.0
    else:
        accuracy = (correct_count / total_count) * 100

    print("\n--- Evaluation Results ---")
    print(f"Total Evaluated: {total_count}")
    print(f"Correct:         {correct_count}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print("------------------------")


if __name__ == "__main__":
    main()
