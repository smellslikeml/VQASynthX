import re
import json
from typing import Dict, Any, List, Tuple

# Inspired by ResearcherBench's Factual Assessment, this module adapts the concept
# to evaluate spatial reasoning claims against a geometric "ground truth" derived
# from the VQASynth pipeline, rather than web-based citations.


def extract_spatial_claims(response: str) -> List[Dict[str, Any]]:
    """
    Extracts spatial reasoning claims from a VLM's response.
    This is a simplified version of ResearcherBench's claim extraction.
    For this demo, we'll use regex to find claims about distances.

    Example Claim: "The <object1> is about <distance> from the <object2>."
    """
    claims = []
    # Regex to find "The [object] is approximately [number] [units] from the [object]"
    pattern = re.compile(
        r"the (.+?) is approximately ([\d\.]+) (\w+) from the (.+?)[\.\n]",
        re.IGNORECASE,
    )
    matches = pattern.finditer(response)

    for i, match in enumerate(matches):
        claims.append(
            {
                "id": f"claim-{i+1}",
                "type": "distance",
                "object1": match.group(1).strip(),
                "value": float(match.group(2)),
                "units": match.group(3).strip(),
                "object2": match.group(4).strip(),
                "text": match.group(0).strip(),
            }
        )
    return claims


def judge_claim(
    claim: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Verifies a single claim against the ground truth scene data.
    This is analogous to ResearcherBench's 'judge model'.

    Returns: (is_supported, reason)
    """
    if claim["type"] == "distance":
        obj1_claim = claim["object1"]
        obj2_claim = claim["object2"]

        # Find the corresponding ground truth entry
        gt_distance_data = None
        for key, gt in ground_truth.get("distances", {}).items():
            # A simple check if both object classes are in the key
            if obj1_claim in key and obj2_claim in key:
                gt_distance_data = gt
                break

        if not gt_distance_data:
            return (
                False,
                f"Ground truth for distance between '{obj1_claim}' and '{obj2_claim}' not found.",
            )

        # Simple tolerance check. Assuming units are meters for simplicity.
        gt_distance = gt_distance_data["distance_meters"]
        claimed_distance = claim["value"]

        tolerance = 0.5  # meters
        if abs(gt_distance - claimed_distance) <= tolerance:
            return (
                True,
                f"Claim supported. Ground truth: {gt_distance:.2f}m, Claimed: {claimed_distance:.2f}m. Within {tolerance}m tolerance.",
            )
        else:
            return (
                False,
                f"Claim not supported. Ground truth: {gt_distance:.2f}m, Claimed: {claimed_distance:.2f}m. Exceeds {tolerance}m tolerance.",
            )

    return False, "Unsupported claim type."


def run_factual_assessment(
    vlm_response: str, ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Orchestrates the factual assessment pipeline.

    1. Extracts claims from the VLM response.
    2. Verifies each claim against the ground truth.
    3. Calculates a 'Faithfulness' score, similar to ResearcherBench.
    """
    claims = extract_spatial_claims(vlm_response)
    if not claims:
        return {
            "faithfulness": 1.0,
            "supported_claims": 0,
            "total_claims": 0,
            "details": [],
        }

    supported_count = 0
    details = []

    for claim in claims:
        is_supported, reason = judge_claim(claim, ground_truth)
        if is_supported:
            supported_count += 1
        details.append(
            {"claim": claim["text"], "supported": is_supported, "reason": reason}
        )

    faithfulness = supported_count / len(claims)

    return {
        "faithfulness": faithfulness,
        "supported_claims": supported_count,
        "total_claims": len(claims),
        "details": details,
    }


def main():
    """
    Main function to run the demonstration.
    """
    print("Running Spatial Reasoning Factual Assessment (inspired by ResearcherBench)")
    print("-" * 70)

    # 1. Define a sample VLM response to a question like:
    # "Describe the distances between the objects on the desk."
    sample_vlm_response = (
        "Based on the image, the laptop is approximately 0.5 meters from the coffee mug.\n"
        "Additionally, the keyboard is approximately 0.8 meters from the laptop."
    )

    # 2. Define the corresponding ground truth from the VQASynth pipeline
    sample_ground_truth = {
        "image_id": "desk_scene_01.jpg",
        "objects": [
            {"id": "obj1", "class": "laptop"},
            {"id": "obj2", "class": "coffee mug"},
            {"id": "obj3", "class": "keyboard"},
        ],
        "distances": {
            "laptop-coffee mug": {"distance_meters": 0.45},
            "laptop-keyboard": {"distance_meters": 0.22},
            "coffee mug-keyboard": {"distance_meters": 0.6},
        },
    }

    print("Sample VLM Response:\n", sample_vlm_response)
    print("\nSample Ground Truth:\n", json.dumps(sample_ground_truth, indent=2))
    print("-" * 70)

    # 3. Run the assessment
    results = run_factual_assessment(sample_vlm_response, sample_ground_truth)

    # 4. Print the results
    print("Assessment Results:\n")
    print(json.dumps(results, indent=2))
    print(f"\nFaithfulness Score: {results['faithfulness']:.2f}")


if __name__ == "__main__":
    main()
