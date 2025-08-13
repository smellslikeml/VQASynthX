import json
import argparse

# --- Evaluation Data & Rubric (Inspired by ResearcherBench's curated questions & rubrics) ---

# A set of challenging spatial reasoning questions that require more than simple object detection.
# In a real scenario, these would be paired with images and run through a VLM.
EVALUATION_QUESTIONS = [
    {
        "id": "scene_01_q1",
        "image_path": "assets/warehouse_sample_1.jpeg",
        "question": "Describe the spatial arrangement of the three largest cardboard boxes relative to the red forklift. Is there a clear path for the forklift to move forward without hitting them?",
        "model_response": "The red forklift is positioned to the left of a stack of brown cardboard boxes. Based on the visible floor space, the forklift appears to have a clear path to move forward. The largest box is on the bottom of the stack.",
    },
    {
        "id": "scene_02_q1",
        "image_path": "assets/warehouse_sample_2.jpeg",
        "question": "Considering the man in the red hat as a reference, what objects are in his immediate foreground and background along his path of movement?",
        "model_response": "In the man's foreground is a wooden pallet with boxes. In his background, there are more shelves and pallets. It seems he is walking past the pallet.",
    },
    {
        "id": "scene_03_q1",
        "image_path": "assets/warehouse_sample_3.jpeg",
        "question": "If the man in the blue shirt were to walk directly towards the camera, would his path intersect with the person sitting on the forklift?",
        "model_response": "It is unclear. The man in blue is far away. The person on the forklift is also there. Their paths might cross.",
    },
]

# A rubric for evaluating the quality of spatial reasoning, inspired by ResearcherBench's expert-designed criteria.
EVALUATION_RUBRIC = {
    "Object Identification": {
        "description": "Correctly identifies all key objects mentioned in the question.",
        "weight": 0.2,
        "scoring_guide": {
            3: "All key objects correctly identified.",
            2: "Most key objects identified, with minor omissions.",
            1: "Significant errors or omissions in object identification.",
            0: "Fails to identify any key objects.",
        },
    },
    "Spatial Relationship Accuracy": {
        "description": "Accurately describes the spatial relationships (e.g., left of, behind, path clear/blocked).",
        "weight": 0.4,
        "scoring_guide": {
            3: "All spatial relationships are described accurately and precisely.",
            2: "Most relationships are correct, but with some ambiguity or minor inaccuracies.",
            1: "Describes relationships with significant errors.",
            0: "Fails to describe spatial relationships.",
        },
    },
    "Compositional Reasoning": {
        "description": "Synthesizes information about multiple objects/relationships to answer a complex question (e.g., path planning).",
        "weight": 0.3,
        "scoring_guide": {
            3: "Demonstrates strong compositional reasoning to answer the core question.",
            2: "Attempts to synthesize information but the conclusion is weak or partially incorrect.",
            1: "Identifies components but fails to synthesize them into a coherent answer.",
            0: "No synthesis or reasoning demonstrated.",
        },
    },
    "Response Clarity and Confidence": {
        "description": "Provides a clear, unambiguous answer and appropriately qualifies uncertainty.",
        "weight": 0.1,
        "scoring_guide": {
            3: "Response is clear, direct, and confidently states conclusions or uncertainties.",
            2: "Response is mostly clear but could be more direct.",
            1: "Response is ambiguous, evasive, or overly uncertain (e.g., 'maybe', 'it is unclear').",
            0: "Response is incoherent.",
        },
    },
}


# --- Evaluation Logic (Simulating ResearcherBench's evaluator) ---


def evaluate_response_with_llm_judge_mock(response_text: str) -> dict:
    """
    Mocks an LLM-as-a-judge evaluation.
    In a real implementation, this would make an API call to a powerful model (like Claude/GPT-4)
    with a prompt that includes the rubric, the question, and the model's response.
    For this example, we use a simple heuristic based on keywords.
    """
    scores = {}
    lower_response = response_text.lower()

    # Heuristic for Object Identification
    if "forklift" in lower_response and "boxes" in lower_response:
        scores["Object Identification"] = 3
    else:
        scores["Object Identification"] = 1

    # Heuristic for Spatial Relationship Accuracy
    if (
        "left of" in lower_response
        or "path" in lower_response
        or "foreground" in lower_response
    ):
        scores["Spatial Relationship Accuracy"] = 3
    elif "past" in lower_response:
        scores["Spatial Relationship Accuracy"] = 2
    else:
        scores["Spatial Relationship Accuracy"] = 1

    # Heuristic for Compositional Reasoning
    if "path" in lower_response and "clear" in lower_response:
        scores["Compositional Reasoning"] = 3
    elif "paths might cross" in lower_response:
        scores["Compositional Reasoning"] = 2
    else:
        scores["Compositional Reasoning"] = 1

    # Heuristic for Clarity
    if "it is unclear" in lower_response or "might" in lower_response:
        scores["Response Clarity and Confidence"] = 1
    else:
        scores["Response Clarity and Confidence"] = 3

    return scores


def run_evaluation(questions: list, rubric: dict) -> dict:
    """
    Runs the evaluation across all questions and returns detailed and summary results.
    """
    results = {"per_question": {}, "summary": {}}
    total_score = 0
    total_possible_score = 0

    print("--- Starting Spatial Reasoning Evaluation ---")

    for item in questions:
        print(f"\nEvaluating Question ID: {item['id']}")
        print(f"  > Question: {item['question']}")
        print(f"  > Model Response: {item['model_response']}")

        # Get scores from our mock LLM judge
        raw_scores = evaluate_response_with_llm_judge_mock(item["model_response"])

        item_weighted_score = 0
        max_item_score = 0
        question_results = {"scores": {}, "weighted_score": 0}

        for criterion, details in rubric.items():
            score = raw_scores.get(criterion, 0)
            weight = details["weight"]
            max_score_for_criterion = max(details["scoring_guide"].keys())

            question_results["scores"][criterion] = f"{score}/{max_score_for_criterion}"
            item_weighted_score += score * weight
            max_item_score += max_score_for_criterion * weight

        question_results["weighted_score"] = round(item_weighted_score, 3)
        question_results["max_possible_score"] = round(max_item_score, 3)
        results["per_question"][item["id"]] = question_results

        total_score += item_weighted_score
        total_possible_score += max_item_score

        print(f"  > Result: Scored {item_weighted_score:.2f} / {max_item_score:.2f}")

    # Calculate summary
    average_score = (
        (total_score / total_possible_score) if total_possible_score > 0 else 0
    )
    results["summary"] = {
        "total_weighted_score": round(total_score, 3),
        "total_possible_score": round(total_possible_score, 3),
        "coverage_score": round(
            average_score, 4
        ),  # Named "coverage_score" to echo ResearcherBench
    }

    print("\n--- Evaluation Complete ---")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run rubric-based evaluation for spatial reasoning."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save the detailed evaluation results.",
    )
    args = parser.parse_args()

    evaluation_results = run_evaluation(EVALUATION_QUESTIONS, EVALUATION_RUBRIC)

    print("\n--- Final Summary ---")
    print(json.dumps(evaluation_results["summary"], indent=2))

    with open(args.output_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\nDetailed results saved to {args.output_file}")
