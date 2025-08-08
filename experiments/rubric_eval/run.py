import json

# This evaluation script is a minimal experiment inspired by the "Rubric Assessment"
# framework from GAIR-NLP/ResearcherBench. The goal is to evaluate the quality of
# a model's response to an "open consulting" style question, focusing on the
# quality of insights rather than simple factual accuracy.
#
# As described in ResearcherBench's README, we define a rubric with weighted
# criteria to score the response. This allows for a more nuanced evaluation
# of the spatial reasoning capabilities developed by the VQASynth pipeline.

# --- Test Data ---
# In a real scenario, this would come from a VLM trained on VQASynth data.
QUESTION = (
    "Considering the attached image of a warehouse, describe the spatial layout "
    "and identify potential challenges for an autonomous forklift navigating from "
    "the red forklift to the stack of brown boxes."
)

MODEL_RESPONSE = (
    "The red forklift is on the left side of the image, near a wall. The brown "
    "cardboard boxes are stacked on a pallet in the center-right. To navigate, "
    "the forklift must first move forward into the main aisle, turn right, and "
    "then proceed towards the boxes. A key challenge is the narrow space between "
    "the stacked pallets and the man walking in the aisle. The forklift must "
    "account for this dynamic obstacle."
)

# --- Rubric Definition (Inspired by ResearcherBench) ---
# Expert-designed criteria with weights to evaluate key insights.
RUBRIC = {
    "criteria": [
        {
            "insight": "Identifies start and end locations",
            "weight": 0.2,
            "keywords": ["red forklift", "brown boxes"],
        },
        {
            "insight": "Describes a plausible path",
            "weight": 0.3,
            "keywords": ["aisle", "turn", "proceed"],
        },
        {
            "insight": "Identifies static obstacles/constraints",
            "weight": 0.2,
            "keywords": ["narrow", "pallets", "space"],
        },
        {
            "insight": "Identifies dynamic obstacles/challenges",
            "weight": 0.3,
            "keywords": ["man", "walking", "dynamic obstacle"],
        },
    ]
}

def evaluate_response(response, rubric):
    """
    Evaluates the response against the rubric.

    This is a simplified version of the ResearcherBench rubric evaluator.
    It checks for the presence of keywords to determine if an insight is covered.
    """
    total_score = 0.0
    evaluation_details = []

    for criterion in rubric["criteria"]:
        insight = criterion["insight"]
        weight = criterion["weight"]
        keywords = criterion["keywords"]

        # Check if all keywords for an insight are present in the response
        is_covered = all(keyword in response.lower() for keyword in keywords)
        score = weight if is_covered else 0.0
        total_score += score
        
        evaluation_details.append({
            "insight": insight,
            "covered": is_covered,
            "score": score,
            "weight": weight,
        })

    return {
        "final_score": total_score,
        "details": evaluation_details,
    }

if __name__ == "__main__":
    print("--- VQASynth Qualitative Spatial Reasoning Evaluation ---")
    print(f"\nQuestion: {QUESTION}")
    print(f"\nModel Response: {MODEL_RESPONSE}")

    results = evaluate_response(MODEL_RESPONSE, RUBRIC)

    print("\n--- Evaluation Results ---")
    print(json.dumps(results, indent=2))
    print(f"\nFinal Weighted Coverage Score: {results['final_score']:.2f} / 1.00")
