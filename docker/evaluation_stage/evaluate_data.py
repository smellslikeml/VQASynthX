import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# Simple data structures inspired by RAVine's nugget concept
class Nugget:
    """A ground-truth fact derived from the scene data."""

    def __init__(
        self,
        type: str,
        value: Any,
        unit: Optional[str] = None,
        objects: Optional[List[str]] = None,
    ):
        self.type = type
        self.value = value
        self.unit = unit
        self.objects = objects or []

    def __repr__(self):
        return f"Nugget(type={self.type}, value={self.value}, unit={self.unit}, objects={self.objects})"


class Claim:
    """A factual claim extracted from the VQA model's generated answer."""

    def __init__(self, type: str, value: Any, unit: Optional[str] = None):
        self.type = type
        self.value = value
        self.unit = unit

    def __repr__(self):
        return f"Claim(type={self.type}, value={self.value}, unit={self.unit})"


def extract_nuggets_from_scene_data(scene_data: Dict[str, Any]) -> List[Nugget]:
    """
    Parses the structured scene data to create a list of ground-truth nuggets.
    This is a simplified example focusing on distances.
    """
    nuggets = []
    if "relationships" in scene_data:
        for rel in scene_data["relationships"]:
            if rel.get("type") == "distance":
                nuggets.append(
                    Nugget(
                        type="distance",
                        value=rel.get("value"),
                        unit=rel.get("unit"),
                        objects=[rel.get("from_object"), rel.get("to_object")],
                    )
                )
    return nuggets


def extract_claims_from_answer(answer_text: str) -> List[Claim]:
    """
    Uses regex to extract factual claims from a generated text answer.
    This is a simplified example focusing on numerical distances.
    """
    claims = []
    # Regex to find a number (int or float) followed by a common unit of distance
    distance_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(meters?|feet|foot|inches|inch|cm|centimeters?)",
        re.IGNORECASE,
    )
    matches = distance_pattern.findall(answer_text)
    for match in matches:
        value, unit = match
        # Normalize unit
        if unit.lower() in ["meter", "meters"]:
            normalized_unit = "meters"
        elif unit.lower() in ["foot", "feet"]:
            normalized_unit = "feet"
        else:
            normalized_unit = unit.lower()  # Basic normalization

        claims.append(Claim(type="distance", value=float(value), unit=normalized_unit))
    return claims


def evaluate_claims(claims: List[Claim], nuggets: List[Nugget]) -> Tuple[int, int, int]:
    """
    Compares extracted claims against ground-truth nuggets.
    Returns: (total_claims, correct_claims, relevant_nuggets)
    """
    if not claims and not nuggets:
        return 0, 0, 0
    if not nuggets:
        return len(claims), 0, 0  # All claims are hallucinated

    correct_claims_count = 0
    # For simplicity, we assume the first relevant nugget is the target
    # A real system would need more complex matching of objects.
    distance_nuggets = [n for n in nuggets if n.type == "distance"]
    distance_claims = [c for c in claims if c.type == "distance"]

    if not distance_nuggets:
        return len(distance_claims), 0, 0

    # Match the first found claim to the first found nugget
    # This is a minimal implementation.
    if distance_claims:
        claim = distance_claims[0]
        nugget = distance_nuggets[0]

        # Check for correctness with a tolerance for floating point issues
        # and potential unit conversions (not implemented here for simplicity)
        if claim.unit == nugget.unit and abs(claim.value - nugget.value) < 0.1:
            correct_claims_count += 1

    return len(distance_claims), correct_claims_count, len(distance_nuggets)


def main():
    """
    Main function to run the evaluation stage.
    Reads synthesized VQA data and structured scene data, then writes out an
    evaluation report.
    """
    # These paths would be mounted in the Docker container
    prompt_data_path = Path(os.environ.get("PROMPT_DATA_DIR", "/data/prompts"))
    scene_data_path = Path(os.environ.get("SCENE_DATA_DIR", "/data/scene_fusion"))
    output_path = Path(os.environ.get("OUTPUT_DIR", "/data/evaluation"))
    output_path.mkdir(exist_ok=True)

    eval_results = []

    prompt_files = list(prompt_data_path.glob("*.json"))
    print(f"Found {len(prompt_files)} prompt files to evaluate.")

    for prompt_file in prompt_files:
        scene_file = scene_data_path / prompt_file.name
        if not scene_file.exists():
            print(
                f"Warning: No corresponding scene file for {prompt_file.name}. Skipping."
            )
            continue

        with open(prompt_file, "r") as f:
            prompt_data = json.load(f)
        with open(scene_file, "r") as f:
            scene_data = json.load(f)

        # Assuming prompt_data is a list of conversations from the VQASynth prompt_stage
        if isinstance(prompt_data, dict) and "conversations" in prompt_data:
            conversations = prompt_data["conversations"]
        elif isinstance(prompt_data, list):
            conversations = prompt_data
        else:
            continue

        for conversation in conversations:
            if conversation.get("from") == "human":
                continue  # Skip questions

            answer_text = conversation.get("value", "")
            nuggets = extract_nuggets_from_scene_data(scene_data)
            claims = extract_claims_from_answer(answer_text)

            total_claims, correct_claims, relevant_nuggets = evaluate_claims(
                claims, nuggets
            )

            precision = correct_claims / total_claims if total_claims > 0 else 0.0
            recall = correct_claims / relevant_nuggets if relevant_nuggets > 0 else 0.0

            eval_results.append(
                {
                    "image_id": scene_data.get("image_id"),
                    "answer": answer_text,
                    "metrics": {
                        "precision": precision,
                        "recall": recall,
                        "correct_claims": correct_claims,
                        "total_claims": total_claims,
                        "relevant_nuggets": relevant_nuggets,
                    },
                    "extracted_claims": [c.__dict__ for c in claims],
                    "ground_truth_nuggets": [n.__dict__ for n in nuggets],
                }
            )

    output_file = output_path / "evaluation_report.jsonl"
    with open(output_file, "w") as f:
        for result in eval_results:
            f.write(json.dumps(result) + "\n")

    print(f"Evaluation complete. Report saved to {output_file}")


if __name__ == "__main__":
    main()
