import argparse
import json
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score


def categorize_question(question_text):
    """
    Categorizes a question as 'Reasoning' or 'Understanding' based on keywords.
    This is a simple heuristic inspired by the need to split evaluation,
    as demonstrated in the MediQAl project.
    Binary questions are treated as 'Reasoning', while others are 'Understanding'.
    """
    reasoning_starters = ("is", "are", "does", "do", "can")
    if question_text.lower().strip().split(" ")[0] in reasoning_starters:
        return "Reasoning"
    return "Understanding"


def main(args):
    # Load model predictions from a JSON file
    # Expected format: [{"id": "...", "prediction": "..."}]
    try:
        with open(args.predictions_file, "r") as f:
            predictions = json.load(f)
        predictions_df = pd.DataFrame(predictions)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading predictions file: {e}")
        return

    # Load the ground truth dataset from Hugging Face Hub
    print(f"Loading dataset '{args.dataset_name}'...")
    dataset = load_dataset(args.dataset_name, split="train")
    ground_truth_df = dataset.to_pandas()

    # Merge predictions with ground truth on the 'id' column
    # VQASynth datasets use 'id' which often corresponds to the image filename
    eval_df = pd.merge(predictions_df, ground_truth_df, on="id")

    if eval_df.empty:
        print(
            "No matching IDs found between predictions and ground truth dataset. Aborting."
        )
        return

    # Apply the categorization logic to create a 'question_type' column
    eval_df["question_type"] = eval_df["question"].apply(categorize_question)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(eval_df["answer"], eval_df["prediction"])

    # Calculate accuracy for 'Understanding' questions
    understanding_df = eval_df[eval_df["question_type"] == "Understanding"]
    understanding_accuracy = 0.0
    if not understanding_df.empty:
        understanding_accuracy = accuracy_score(
            understanding_df["answer"], understanding_df["prediction"]
        )

    # Calculate accuracy for 'Reasoning' questions
    reasoning_df = eval_df[eval_df["question_type"] == "Reasoning"]
    reasoning_accuracy = 0.0
    if not reasoning_df.empty:
        reasoning_accuracy = accuracy_score(
            reasoning_df["answer"], reasoning_df["prediction"]
        )

    # Print the results in a structured format
    print("\n--- VQA Performance Evaluation ---")
    print(f"Dataset: {args.dataset_name}")
    print(f"Predictions: {args.predictions_file}")
    print("-" * 45)
    print(f"| {'Category':<15} | {'Accuracy':<12} | {'Count':<10} |")
    print(f"|-----------------|--------------|------------|")
    print(f"| {'Overall':<15} | {overall_accuracy:<12.4f} | {len(eval_df):<10} |")
    print(
        f"| {'Understanding':<15} | {understanding_accuracy:<12.4f} | {len(understanding_df):<10} |"
    )
    print(
        f"| {'Reasoning':<15} | {reasoning_accuracy:<12.4f} | {len(reasoning_df):<10} |"
    )
    print("-" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VQA model performance with a split between understanding and reasoning questions."
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to the JSON file containing model predictions. Expected format: [{'id': ..., 'prediction': ...}]",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="remyxai/OpenSpaces_MC_R1",
        help="Name of the Hugging Face dataset for ground truth.",
    )

    args = parser.parse_args()
    main(args)
