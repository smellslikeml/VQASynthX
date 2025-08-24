import json
from typing import List, Dict, Any

# Mock scene data simulating the output of an object detection/recognition pipeline
# where detections can have multiple, scored labels, representing a "belief state".
# This is inspired by the probabilistic piece types in Belief-SG.
MOCK_SCENE_DATA = {
    "image_id": "scene_001",
    "objects": [
        {
            "id": "obj_1",
            "bbox": [100, 150, 250, 400],  # [x_min, y_min, x_max, y_max]
            "potential_labels": [
                {"label": "armchair", "score": 0.85},
                {"label": "recliner", "score": 0.10},
            ],
            "center_point": [175, 275],  # [x, y]
        },
        {
            "id": "obj_2",
            "bbox": [450, 200, 550, 300],
            "potential_labels": [
                {"label": "side table", "score": 0.92},
                {"label": "stool", "score": 0.05},
            ],
            "center_point": [500, 250],
        },
        {
            "id": "obj_3",
            "bbox": [600, 50, 780, 450],
            "potential_labels": [
                {"label": "floor lamp", "score": 0.98},
            ],
            "center_point": [690, 250],
        },
    ],
}


def get_primary_label(obj: Dict[str, Any]) -> str:
    """Returns the most likely label for an object."""
    return obj["potential_labels"][0]["label"]


def generate_disambiguation_qa(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates questions asking to disambiguate an object's identity.
    Example: "In the image, is the object better described as an armchair or a recliner?"
    """
    qa_pairs = []
    if len(obj["potential_labels"]) > 1:
        label1 = obj["potential_labels"][0]["label"]
        label2 = obj["potential_labels"][1]["label"]
        question = f"Considering the object at bounding box {obj['bbox']}, is it an {label1} or a {label2}?"
        # The "correct" answer is the one with the higher score.
        answer = f"It is an {label1}."
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs


def generate_conditional_qa(scene_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates questions based on a hypothetical premise about an ambiguous object.
    Inspired by PAPER REASONING: "constraint-based beliefs can be as effective as probabilistic ones".
    We constrain the world with an "Assuming..." clause.
    Example: "Assuming the object is a recliner, is it to the left of the side table?"
    """
    qa_pairs = []
    objects = scene_data["objects"]
    if len(objects) < 2:
        return []

    # Find an ambiguous object to use for the hypothesis
    for i, obj1 in enumerate(objects):
        if len(obj1["potential_labels"]) > 1:
            # The hypothetical label is the second most likely one
            hypothetical_label = obj1["potential_labels"][1]["label"]

            # Find another object to relate it to
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue

                # Use primary label for the second object
                obj2_label = get_primary_label(obj2)

                # Basic spatial relationship based on center point x-coordinate
                is_left = obj1["center_point"][0] < obj2["center_point"][0]
                relation = "to the left of" if is_left else "to the right of"

                question = f"Assuming the object at {obj1['bbox']} is a {hypothetical_label}, is it {relation} the {obj2_label}?"
                answer = f"Yes, based on the positions in the image, the object assumed to be a {hypothetical_label} is indeed {relation} the {obj2_label}."
                qa_pairs.append({"question": question, "answer": answer})

                # Only generate one conditional question per ambiguous object for simplicity
                break
    return qa_pairs


def generate_vqa_from_belief_state(scene_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Main function to generate various VQA pairs from a scene with belief states.
    """
    all_qa_pairs = []

    # 1. Generate disambiguation questions for each ambiguous object
    for obj in scene_data["objects"]:
        all_qa_pairs.extend(generate_disambiguation_qa(obj))

    # 2. Generate conditional/hypothetical questions for the scene
    all_qa_pairs.extend(generate_conditional_qa(scene_data))

    return all_qa_pairs


if __name__ == "__main__":
    print("--- VQA Synthesis from Ambiguous Belief States ---")
    print(
        "This script demonstrates generating VQA pairs that handle uncertainty in object recognition."
    )
    print(
        "This is inspired by the belief state management in imperfect-information games like Belief-SG.\n"
    )

    generated_qa = generate_vqa_from_belief_state(MOCK_SCENE_DATA)

    print(f"Generated {len(generated_qa)} Q&A pairs from the mock scene:")
    print(json.dumps(generated_qa, indent=2))
