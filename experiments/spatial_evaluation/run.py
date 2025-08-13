import json
import re

# Inspired by ResearcherBench's rubric assessment for evaluating complex reasoning.
# This script adapts the concept to evaluate VLM spatial reasoning on VQASynth's domain.
# The ground truth data (scene_understanding) acts as the 'expert knowledge'
# and the rubric defines the key insights we expect the model to capture.

# 1. TEST DATA: Questions and Ground Truth (simulating VQASynth output)
EVALUATION_SET = [
    {
        "id": "warehouse_sample_1",
        "image_path": "assets/warehouse_sample_1.jpeg",
        "question": "Does the red forklift appear on the left side of the brown cardboard boxes? How far is it?",
        "scene_understanding": {
            "objects": [
                {"id": "obj1", "desc": "red forklift", "center_x": 200},
                {"id": "obj2", "desc": "brown cardboard boxes", "center_x": 600},
            ],
            "distance_meters": 5.2,
        },
    },
    {
        "id": "warehouse_sample_2",
        "image_path": "assets/warehouse_sample_2.jpeg",
        "question": "How close is the man in the red hat to the wooden pallet with boxes?",
        "scene_understanding": {
            "objects": [
                {"id": "obj1", "desc": "man in red hat", "center_x": 450},
                {"id": "obj2", "desc": "wooden pallet with boxes", "center_x": 550},
            ],
            "distance_meters": 1.5,
        },
    },
]

# 2. RUBRIC DEFINITION: Inspired by ResearcherBench's expert-designed rubrics.
# Each criterion is a function that returns a score (0 or 1).
# This structure allows for weighted scoring, similar to ResearcherBench.
RUBRIC = {
    "orientation_correct": {
        "weight": 2,
        "eval_func": lambda response, scene: (
            (
                "left" in response.lower()
                and scene["objects"][0]["center_x"] < scene["objects"][1]["center_x"]
            )
            or (
                "right" in response.lower()
                and scene["objects"][0]["center_x"] > scene["objects"][1]["center_x"]
            )
        ),
    },
    "distance_mentioned": {
        "weight": 1,
        "eval_func": lambda response, scene: bool(
            re.search(r"\d+(\.\d+)?\s*(m|meters|feet)", response.lower())
        ),
    },
    "distance_accurate": {
        "weight": 3,
        "eval_func": lambda response, scene: (
            # Extracts the first number found in the response
            (num_match := re.search(r"\d+(\.\d+)?", response))
            and abs(float(num_match.group(0)) - scene["distance_meters"])
            < 1.0  # Tolerance of 1 meter
        ),
    },
}

# This specialized rubric is for questions that don't ask about orientation
DISTANCE_ONLY_RUBRIC = {k: v for k, v in RUBRIC.items() if k != "orientation_correct"}


# 3. HYPOTHETICAL MODEL RESPONSES (to be replaced by actual model output)
MODEL_RESPONSES = {
    "model_A_good": {
        "warehouse_sample_1": "Yes, the red forklift is on the left of the boxes. It is approximately 5 meters away.",
        "warehouse_sample_2": "The man in the red hat is very close to the pallet, I'd estimate about 1.8 meters.",
    },
    "model_B_poor": {
        "warehouse_sample_1": "The red forklift is near the brown boxes.",
        "warehouse_sample_2": "The man is standing on the right side of the pallet.",
    },
}


def evaluate_response(response, question_data):
    """
    Evaluates a single response against the rubric, inspired by ResearcherBench's methodology.
    """
    scene = question_data["scene_understanding"]
    total_score = 0
    max_score = 0

    # Select appropriate rubric based on question type
    rubric_to_use = RUBRIC
    if "how close" in question_data["question"].lower():
        rubric_to_use = DISTANCE_ONLY_RUBRIC

    results = {}
    for key, criterion in rubric_to_use.items():
        weight = criterion["weight"]
        eval_func = criterion["eval_func"]

        # Use a try-except block to handle parsing errors or missing info in response
        try:
            is_met = eval_func(response, scene)
        except (ValueError, TypeError, IndexError):
            is_met = False

        score = 1 if is_met else 0
        total_score += score * weight
        max_score += weight
        results[key] = {"met": bool(is_met), "score": score * weight}

    final_score = total_score / max_score if max_score > 0 else 0
    return {"final_score": final_score, "breakdown": results}


def main():
    """
    Main function to run the evaluation for all models.
    """
    print(
        "Running Spatial Reasoning Evaluation (inspired by ResearcherBench Rubric Assessment)"
    )

    all_results = {}
    for model_name, responses in MODEL_RESPONSES.items():
        print(f"\n--- Evaluating Model: {model_name} ---")
        model_scores = []
        for question_data in EVALUATION_SET:
            q_id = question_data["id"]
            if q_id in responses:
                response_text = responses[q_id]
                result = evaluate_response(response_text, question_data)
                model_scores.append(result["final_score"])
                print(f"Question ID: {q_id}")
                print(f"  Response: '{response_text}'")
                print(f"  Score: {result['final_score']:.2f}")
                print(f"  Breakdown: {json.dumps(result['breakdown'], indent=2)}")

        avg_score = sum(model_scores) / len(model_scores) if model_scores else 0
        all_results[model_name] = {"average_score": avg_score}
        print(f"--- Average Score for {model_name}: {avg_score:.2f} ---")

    print("\n--- Evaluation Summary ---")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
