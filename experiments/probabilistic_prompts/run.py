import json
import numpy as np
import os


def add_uncertainty(point_estimate, uncertainty_factor=0.05):
    """
    Simulates uncertainty by treating a point estimate as the mean of a
    normal distribution and calculating a standard deviation.
    This concept is inspired by UQGNN, which outputs a distribution
    rather than a single value.

    Args:
        point_estimate (float): The calculated value (e.g., distance).
        uncertainty_factor (float): Multiplier to determine std dev from the mean.

    Returns:
        dict: A dictionary with 'mean' and 'std_dev'.
    """
    mean = float(point_estimate)
    std_dev = float(abs(point_estimate * uncertainty_factor))
    return {"mean": mean, "std_dev": std_dev}


def generate_probabilistic_qa(obj1, obj2, distance_measurement):
    """
    Generates VQA pairs that express uncertainty, based on a simulated
    probabilistic measurement.

    Args:
        obj1 (str): Name of the first object.
        obj2 (str): Name of the second object.
        distance_measurement (dict): Dict containing 'mean' and 'std_dev'.

    Returns:
        list: A list of conversation dictionaries.
    """
    mean_dist = round(distance_measurement["mean"], 2)
    std_dev = round(distance_measurement["std_dev"], 2)

    # Simple check for confidence level based on relative error
    relative_error = std_dev / mean_dist if mean_dist > 0 else 0
    if relative_error < 0.05:
        confidence_level = "very high"
    elif relative_error < 0.1:
        confidence_level = "high"
    else:
        confidence_level = "moderate"

    conversations = [
        {
            "from": "human",
            "value": f"What is the approximate distance between the {obj1} and the {obj2}?",
        },
        {
            "from": "gpt",
            "value": f"The distance between the {obj1} and the {obj2} is approximately {mean_dist} meters.",
        },
        {
            "from": "human",
            "value": "Can you be more specific about the measurement uncertainty?",
        },
        {
            "from": "gpt",
            "value": f"My estimate is centered at {mean_dist} meters with a standard deviation of {std_dev} meters. This represents a {confidence_level} level of confidence in the measurement.",
        },
    ]
    return conversations


def main():
    """
    Main function to run the experiment.
    """
    print("Running probabilistic prompt generation experiment...")

    # 1. Mock input data from a hypothetical 'scene_fusion' stage.
    # This data would typically be produced by the VQASynth pipeline.
    mock_scene_data = [
        {"obj1": "red forklift", "obj2": "cardboard boxes", "distance_meters": 3.5},
        {"obj1": "man in red hat", "obj2": "wooden pallet", "distance_meters": 1.2},
        {"obj1": "person", "obj2": "desk", "distance_meters": 8.1},
    ]

    output_vqa_data = []

    # 2. Process each scene to generate probabilistic VQA pairs.
    for scene in mock_scene_data:
        # The core idea from UQGNN: represent the measurement as a distribution.
        probabilistic_distance = add_uncertainty(
            scene["distance_meters"], uncertainty_factor=0.08
        )

        qa_pairs = generate_probabilistic_qa(
            scene["obj1"], scene["obj2"], probabilistic_distance
        )

        output_vqa_data.append(
            {
                "id": f"scene_{len(output_vqa_data)}",
                "image": "path/to/image.jpg",  # Placeholder image path
                "conversations": qa_pairs,
            }
        )

    # 3. Save the output to a file for inspection.
    output_dir = "output/probabilistic_prompts"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vqa_data.json")

    with open(output_path, "w") as f:
        json.dump(output_vqa_data, f, indent=2)

    print(f"Successfully generated {len(output_vqa_data)} VQA samples.")
    print(f"Output saved to: {output_path}")
    print("\n--- Sample 1 ---")
    print(json.dumps(output_vqa_data[0], indent=2))
    print("------------------")


if __name__ == "__main__":
    main()
