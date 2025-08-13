import argparse
import json
import os
import statistics
from typing import Any, Dict, List

# This script requires the 'openai' library.
# Please install it if you haven't: pip install openai

# Lazy import OpenAI to provide a better error message if not installed.
try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' library not found. Please run 'pip install openai'.")
    exit(1)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class VQAEvaluator:
    """
    Evaluates VQA model predictions against ground truth using an LLM-as-judge,
    inspired by the nugget-based evaluation in the RAVine framework.
    """

    EVAL_PROMPT_TEMPLATE = """
You are an impartial and expert evaluator for Vision-Question-Answering systems. Your task is to assess a predicted answer against a ground-truth answer for a given question. The question is about an image, but you will only be given the text.

**Question:**
{question}

**Ground-Truth Answer:**
{ground_truth_answer}

**Predicted Answer:**
{predicted_answer}

Please evaluate the "Predicted Answer" based on the following criteria, assuming the "Ground-Truth Answer" is perfect and complete.
1.  **Completeness**: How much of the key information from the ground-truth answer is present in the predicted answer? A score of 1.0 means all key information is present. A score of 0.0 means no key information is present.
2.  **Precision**: Is all the information in the predicted answer correct and relevant, according to the ground-truth? A score of 1.0 means all information is correct and relevant. A score of 0.0 means all information is incorrect or irrelevant.

Provide your evaluation *only* as a JSON object with two keys: "completeness_score" and "precision_score". The scores must be floats between 0.0 and 1.0. Do not add any other text or explanation.

**Evaluation JSON:**
"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def evaluate_single_pair(
        self, question: str, ground_truth_answer: str, predicted_answer: str
    ) -> Dict[str, float]:
        """
        Uses the LLM to evaluate a single prediction against its ground truth.
        """
        prompt = self.EVAL_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth_answer=ground_truth_answer,
            predicted_answer=predicted_answer,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            scores = json.loads(content)

            # Validate scores
            if not all(
                isinstance(scores.get(k), (int, float))
                and 0.0 <= scores.get(k, -1) <= 1.0
                for k in ["completeness_score", "precision_score"]
            ):
                print(f"Warning: Invalid scores received: {scores}. Defaulting to 0.")
                return {"completeness_score": 0.0, "precision_score": 0.0}

            return {
                "completeness_score": float(scores["completeness_score"]),
                "precision_score": float(scores["precision_score"]),
            }

        except Exception as e:
            print(f"Error during LLM evaluation for question '{question[:30]}...': {e}")
            return {"completeness_score": 0.0, "precision_score": 0.0}

    def run(
        self,
        ground_truth_data: List[Dict[str, Any]],
        prediction_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Runs the evaluation across all data, aggregates scores, and returns metrics.
        """
        gt_map = {item["id"]: item for item in ground_truth_data}
        pred_map = {item["id"]: item for item in prediction_data}

        all_scores = []
        item_ids = sorted(list(gt_map.keys()))

        if not item_ids:
            print("Warning: No ground truth data found.")
            return {}

        for item_id in item_ids:
            if item_id not in pred_map:
                print(f"Warning: No prediction found for ID {item_id}. Skipping.")
                continue

            gt_item = gt_map[item_id]
            pred_item = pred_map[item_id]

            print(f"Evaluating ID: {item_id}")
            scores = self.evaluate_single_pair(
                question=gt_item["question"],
                ground_truth_answer=gt_item["answer"],
                predicted_answer=pred_item["answer"],
            )
            all_scores.append(scores)

        if not all_scores:
            print("Error: No items could be evaluated.")
            return {}

        # Aggregate metrics, inspired by RAVine's report quality metrics
        avg_completeness = statistics.mean(s["completeness_score"] for s in all_scores)
        avg_precision = statistics.mean(s["precision_score"] for s in all_scores)

        # F1-score for a combined metric
        f1_score = 0
        if (avg_completeness + avg_precision) > 0:
            f1_score = (
                2
                * (avg_completeness * avg_precision)
                / (avg_completeness + avg_precision)
            )

        results = {
            "evaluation_metrics": {
                "completeness": round(avg_completeness, 4),
                "precision": round(avg_precision, 4),
                "f1_score": round(f1_score, 4),
            },
            "evaluated_count": len(all_scores),
            "total_ground_truth_count": len(gt_map),
        }
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQA predictions using RAVine-inspired metrics."
    )
    parser.add_argument(
        "--ground-truth-path",
        type=str,
        required=True,
        help="Path to the ground truth JSONL file. Each line must be a JSON object with 'id', 'question', and 'answer' keys.",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Path to the predictions JSONL file. Each line must be a JSON object with 'id' and 'answer' keys.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results JSON file.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="The OpenAI model to use for evaluation.",
    )

    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: OpenAI API key must be provided via the --api-key argument or by setting the OPENAI_API_KEY environment variable."
        )
        exit(1)

    print("Loading data...")
    ground_truth_data = load_jsonl(args.ground_truth_path)
    prediction_data = load_jsonl(args.predictions_path)

    print("Initializing evaluator...")
    evaluator = VQAEvaluator(api_key=args.api_key, model=args.model)

    print("Running evaluation...")
    results = evaluator.run(ground_truth_data, prediction_data)

    print("\n--- Evaluation Complete ---")
    print(json.dumps(results, indent=2))

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
