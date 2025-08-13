import json
import argparse
import os
import re
import numpy as np
from typing import Dict, Any, List, Tuple


# This is a simplified distance calculation for demonstration purposes.
# A real implementation would handle 3D coordinates properly.
def calculate_euclidean_distance(
    coords1: Tuple[float, float, float], coords2: Tuple[float, float, float]
) -> float:
    """Calculates Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(coords1) - np.array(coords2))


def parse_distance_from_text(text: str) -> float | None:
    """Extracts a numerical distance from a string (e.g., 'about 5.2 meters')."""
    # Simple regex to find numbers (including decimals)
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def verify_spatial_claim(
    qa_pair: Dict[str, Any], scene_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verifies the factual consistency of a spatial QA pair against scene data.
    Inspired by ResearcherBench's factual assessment, we treat the VQA answer as a 'claim'
    and the scene data as the 'source' to verify against.
    """
    # Assumes LLaVA conversation format
    if not qa_pair.get("conversations") or len(qa_pair["conversations"]) < 2:
        return {
            "is_verifiable": False,
            "is_consistent": None,
            "consistency_score": None,
        }

    question = qa_pair["conversations"][0].get("value", "")
    answer = qa_pair["conversations"][1].get("value", "")

    # For this minimal example, we only handle simple distance questions.
    if "how far" not in question.lower() and "how close" not in question.lower():
        return {
            "is_verifiable": False,
            "is_consistent": None,
            "consistency_score": None,
        }

    # Extract object names from the question (simplified)
    # A real implementation would need more robust entity extraction.
    objects_in_scene = scene_data.get("objects", {}).keys()
    object_names = [obj for obj in objects_in_scene if obj in question]
    if len(object_names) < 2:
        return {
            "is_verifiable": False,
            "is_consistent": None,
            "consistency_score": None,
        }

    # Get object coordinates from the scene data
    obj1_coords = scene_data["objects"].get(object_names[0], {}).get("coordinates_3d")
    obj2_coords = scene_data["objects"].get(object_names[1], {}).get("coordinates_3d")
    if not obj1_coords or not obj2_coords:
        return {
            "is_verifiable": False,
            "is_consistent": None,
            "consistency_score": None,
        }

    # Calculate ground truth distance from scene data
    true_distance = calculate_euclidean_distance(tuple(obj1_coords), tuple(obj2_coords))

    # Extract claimed distance from the model's answer
    claimed_distance = parse_distance_from_text(answer)
    if claimed_distance is None:
        return {
            "is_verifiable": True,
            "is_consistent": False,
            "consistency_score": 0.0,
            "reason": "Could not parse distance from answer.",
        }

    # Compare distances and calculate a simple consistency score
    # We allow a 15% tolerance for imprecision in language ("about 5 meters")
    tolerance = 0.15
    is_consistent = (
        abs(true_distance - claimed_distance) / true_distance <= tolerance
        if true_distance > 0
        else claimed_distance == 0
    )

    # Score is 1.0 if consistent, 0.0 otherwise.
    consistency_score = 1.0 if is_consistent else 0.0

    return {
        "is_verifiable": True,
        "is_consistent": is_consistent,
        "consistency_score": consistency_score,
        "details": {
            "true_distance": true_distance,
            "claimed_distance": claimed_distance,
            "object_1": object_names[0],
            "object_2": object_names[1],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="""
        Verify factual consistency of generated VQA data.
        This script adapts the 'Factual Assessment' concept from ResearcherBench
        to the VQASynth pipeline, ensuring that generated spatial claims are
        grounded in the reconstructed 3D scene data.
    """
    )
    parser.add_argument(
        "--vqa_data_path",
        type=str,
        required=True,
        help="Path to the JSON file with generated VQA pairs.",
    )
    parser.add_argument(
        "--scene_data_path",
        type=str,
        required=True,
        help="Path to the JSON file with fused scene data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the verified VQA data.",
    )
    args = parser.parse_args()

    # Load data
    with open(args.vqa_data_path, "r") as f:
        vqa_data = json.load(f)
    with open(args.scene_data_path, "r") as f:
        # Assuming a single scene data file corresponds to a single VQA data file
        scene_data = json.load(f)

    verified_vqa_data = []
    for qa_pair in vqa_data:
        verification_result = verify_spatial_claim(qa_pair, scene_data)
        qa_pair["factual_consistency"] = verification_result
        verified_vqa_data.append(qa_pair)

    # Save output
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(verified_vqa_data, f, indent=2)

    print(
        f"Factual consistency check complete. Verified data saved to {args.output_path}"
    )


if __name__ == "__main__":
    main()
