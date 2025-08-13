import json
import argparse
import os
from typing import List, Dict, Any

# ==============================================================================
# Inspired by ResearcherBench's dual evaluation framework, this script adapts
# the rubric-based assessment methodology to evaluate the spatial reasoning
# capabilities of a Vision Language Model (VLM).
#
# SOURCE: https://github.com/GAIR-NLP/ResearcherBench
#
# The core idea is to move beyond simple accuracy and evaluate the quality of
# a model's reasoning process against a structured, expert-defined rubric.
# In VQASynth, this allows us to quantitatively measure if models trained on
# our synthetic data are actually improving in their spatial understanding.
#
# This script implements a minimal version of this concept:
# 1. Defines a rubric for evaluating spatial reasoning answers.
# 2. Loads a VQA dataset (questions and ground-truth).
# 3. Loads corresponding model-generated responses.
# 4. Scores each response against the rubric.
# 5. Aggregates and reports the scores.
# ==============================================================================

# Define the rubric for spatial reasoning, inspired by ResearcherBench's
# expert-designed rubrics. Each criterion has a weight.
SPATIAL_REASONING_RUBRIC = {
    "object_identification": {
        "description": "Correctly identifies the key objects mentioned in the query.",
        "weight": 3,
        "levels": {
            0: "Failed to identify any key objects.",
            1: "Identified some, but not all, key objects correctly.",
            2: "Identified all key objects correctly.",
        },
    },
    "relationship_accuracy": {
        "description": "Accurately describes the spatial relationship (e.g., left of, behind, near).",
        "weight": 4,
        "levels": {
            0: "Incorrectly describes the relationship.",
            1: "Partially correct or ambiguous description.",
            2: "Clear and accurate description of the relationship.",
        },
    },
    "metric_estimation": {
        "description": "Provides a plausible metric estimate if requested (e.g., 'about 5 meters').",
        "weight": 2,
        "levels": {
            0: "Failed to provide a metric estimate or the estimate is wildly inaccurate.",
            1: "Provided a plausible but imprecise estimate.",
            2: "Provided a plausible and reasonably accurate estimate.",
        },
    },
    "reasoning_clarity": {
        "description": "The explanation or reasoning process is clear and logical.",
        "weight": 1,
        "levels": {
            0: "No reasoning provided or the reasoning is nonsensical.",
            1: "Reasoning is provided but is hard to follow or contains logical gaps.",
            2: "Reasoning is clear, step-by-step, and logical (demonstrates CoT).",
        },
    },
}


def load_json(path: str) -> List[Dict[str, Any]]:
    """Loads a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def evaluate_response_with_llm(
    question: str, response: str, ground_truth: str, rubric: Dict
) -> Dict[str, Any]:
    """
    Simulates an LLM-as-a-judge evaluation based on the rubric.
    In a real implementation, this would call an API (like in ResearcherBench).
    For this minimal example, we use a simplified, rule-based simulation.
    """
    print(f"\n--- Evaluating ---")
    print(f"Q: {question}")
    print(f"A_model: {response}")
    print(f"A_gt: {ground_truth}")

    scores = {}
    # Simulate object identification check
    if "forklift" in response.lower() and "boxes" in response.lower():
        scores["object_identification"] = 2
    else:
        scores["object_identification"] = 0

    # Simulate relationship accuracy check
    if "left of" in response.lower() and "left of" in ground_truth.lower():
        scores["relationship_accuracy"] = 2
    elif "near" in response.lower() and "close to" in ground_truth.lower():
        scores["relationship_accuracy"] = 2
    else:
        scores["relationship_accuracy"] = 0

    # Simulate metric estimation
    if "meters" in response or "feet" in response:
        scores["metric_estimation"] = 1  # Simple check for this demo
    else:
        scores["metric_estimation"] = 0

    # Simulate reasoning clarity
    if "because" in response.lower() or "step 1" in response.lower():
        scores["reasoning_clarity"] = 2
    else:
        scores["reasoning_clarity"] = 1

    print(f"Scores: {scores}")
    return scores


def calculate_weighted_score(scores: Dict, rubric: Dict) -> float:
    """Calculates the final weighted score for a single response."""
    total_score = 0
    total_weight = 0
    max_possible_score = 0

    for criterion, level in scores.items():
        weight = rubric[criterion]["weight"]
        max_level = max(rubric[criterion]["levels"].keys())

        total_score += level * weight
        total_weight += weight
        max_possible_score += max_level * weight

    # Normalize the score to be between 0 and 1
    return total_score / max_possible_score if max_possible_score > 0 else 0


def main(args):
    """Main evaluation loop."""
    # Load data (similar to ResearcherBench loading user_data)
    questions_data = load_json(args.questions_file)
    model_responses = load_json(args.responses_file)

    # Create a mapping from question ID to ground truth for easy lookup
    ground_truth_map = {
        item["id"]: item["conversations"][1]["value"] for item in questions_data
    }

    # In VQASynth, a question might be the first turn, and the GT answer the second.
    # We'll map the model responses to the questions by ID.
    response_map = {item["id"]: item["response"] for item in model_responses}

    all_scores = []
    for item in questions_data:
        q_id = item["id"]
        question_text = item["conversations"][0]["value"]

        if q_id not in response_map:
            print(f"Warning: No response found for question ID {q_id}. Skipping.")
            continue

        model_response = response_map[q_id]
        ground_truth = ground_truth_map[q_id]

        # Evaluate the response using the rubric
        # This step is analogous to ResearcherBench's `evaluator.py`
        scores = evaluate_response_with_llm(
            question_text, model_response, ground_truth, SPATIAL_REASONING_RUBRIC
        )

        # Calculate the final score for this response
        # Analogous to ResearcherBench's score aggregation
        weighted_score = calculate_weighted_score(scores, SPATIAL_REASONING_RUBRIC)

        all_scores.append(
            {
                "id": q_id,
                "question": question_text,
                "response": model_response,
                "scores": scores,
                "weighted_score": weighted_score,
            }
        )

    # Aggregate results, similar to ResearcherBench's report generation
    final_avg_score = sum(s["weighted_score"] for s in all_scores) / len(all_scores)

    print("\n\n--- Evaluation Summary ---")
    print(f"Model evaluated: {os.path.basename(args.responses_file)}")
    print(f"Total questions evaluated: {len(all_scores)}")
    print(f"Average Weighted Rubric Score: {final_avg_score:.4f}")
    print("------------------------")

    # Save detailed results to a file
    output_path = os.path.join(args.output_dir, "evaluation_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"Detailed report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM spatial reasoning using a rubric-based approach inspired by ResearcherBench."
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        required=True,
        help="Path to the JSON file containing VQA questions and ground-truth answers.",
    )
    parser.add_argument(
        "--responses_file",
        type=str,
        required=True,
        help="Path to the JSON file containing model responses, mapping to question IDs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/spatial_eval",
        help="Directory to save the evaluation report.",
    )
    args = parser.parse_args()
    main(args)
