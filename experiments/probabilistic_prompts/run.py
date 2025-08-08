import json
import argparse
import random
import numpy as np


def add_uncertainty_to_scene(scene_data, noise_factor=0.1):
    """Artificially introduces uncertainty to distance metrics in a scene."""
    if 'relations' not in scene_data:
        return scene_data

    for relation in scene_data['relations']:
        if 'distance_meters' in relation:
            original_distance = relation['distance_meters']
            if original_distance > 0:
                std_dev = original_distance * noise_factor
                # Replace the scalar distance with a probabilistic representation
                relation['distance_probabilistic'] = {
                    'mean': original_distance,
                    'std_dev': round(std_dev, 3)
                }
    return scene_data


def generate_probabilistic_qa(scene_data):
    """Generates QA pairs that reflect uncertainty."""
    qa_pairs = []
    if 'relations' not in scene_data or not scene_data['relations']:
        return qa_pairs

    for relation in scene_data['relations']:
        if 'distance_probabilistic' in relation:
            obj1_name = relation['obj1_name']
            obj2_name = relation['obj2_name']
            dist_data = relation['distance_probabilistic']

            # Generate a question about the estimated distance
            question = f"What is the estimated distance between the {obj1_name} and the {obj2_name}? Please consider the uncertainty of the measurement."

            # Generate a Chain-of-Thought style answer reflecting the uncertainty
            mean_dist = dist_data['mean']
            std_dev = dist_data['std_dev']
            confidence = 'high' if (std_dev / mean_dist) < 0.1 else 'medium' if (std_dev / mean_dist) < 0.2 else 'low'

            answer = (
                f"Thinking: The system estimates the positions of the {obj1_name} and the {obj2_name} from a 2D image, which introduces some uncertainty. "
                f"The calculated mean distance is {mean_dist:.2f} meters, with a standard deviation of {std_dev:.2f} meters. "
                f"This implies a {confidence} confidence level in the measurement.\n"
                f"Final Answer: The estimated distance between the {obj1_name} and the {obj2_name} is approximately {mean_dist:.2f} meters. This estimate is made with {confidence} confidence."
            )
            
            qa_pairs.append({
                'id': f"{scene_data['image_id']}_rel_{relation['id']}",
                'image': scene_data['image_path'],
                'conversations': [
                    {'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}
                ]
            })

    return qa_pairs


def main():
    parser = argparse.ArgumentParser(description='Generate probabilistic VQA prompts from scene fusion data.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file from the scene fusion stage.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output VQA dataset.')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        scene_data = json.load(f)

    # Process a single scene or a list of scenes
    if isinstance(scene_data, list):
        all_qa_pairs = []
        for scene in scene_data:
            scene_with_uncertainty = add_uncertainty_to_scene(scene)
            all_qa_pairs.extend(generate_probabilistic_qa(scene_with_uncertainty))
        output_data = all_qa_pairs
    else:
        scene_with_uncertainty = add_uncertainty_to_scene(scene_data)
        output_data = generate_probabilistic_qa(scene_with_uncertainty)

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully generated {len(output_data)} probabilistic QA pairs to {args.output_file}")

if __name__ == '__main__':
    main()
