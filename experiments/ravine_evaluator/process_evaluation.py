import argparse
import json
import numpy as np
from itertools import combinations

# Agents are defined by their labels in the scene data.
# This list can be expanded based on the dataset.
AGENT_LABELS = ['person', 'robot', 'forklift', 'man', 'woman']

# Define a threshold for cooperation in meters. If two agents are within this distance,
# they are considered capable of a simple cooperative task (e.g., passing an object).
COOPERATION_THRESHOLD_METERS = 2.5

def calculate_distance(obj1_coords, obj2_coords):
    """Calculate the Euclidean distance between two objects using their 3D coordinates."""
    p1 = np.array(obj1_coords)
    p2 = np.array(obj2_coords)
    return np.linalg.norm(p1 - p2)

def process_scene(scene_data):
    """Processes a single scene to find cooperative agent pairs."""
    agents = []
    if 'objects' not in scene_data:
        return 0, 0

    for obj in scene_data['objects']:
        # Check if the object label corresponds to a defined agent type
        if any(agent_label in obj.get('label', '').lower() for agent_label in AGENT_LABELS):
            if 'center_3d' in obj and obj['center_3d'] is not None:
                agents.append(obj)

    if len(agents) < 2:
        return 0, 0

    potential_pairs = 0
    successful_pairs = 0

    for agent1, agent2 in combinations(agents, 2):
        potential_pairs += 1
        distance = calculate_distance(agent1['center_3d'], agent2['center_3d'])

        if distance <= COOPERATION_THRESHOLD_METERS:
            successful_pairs += 1

    return potential_pairs, successful_pairs

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VQASynth scene data for multi-agent cooperation potential.'
    )
    parser.add_argument(
        '--input_data', 
        type=str, 
        required=True, 
        help='Path to the scene fusion JSON file from the VQASynth pipeline.'
    )
    parser.add_argument(
        '--output_metrics', 
        type=str, 
        required=True, 
        help='Path to save the output evaluation metrics as a JSON file.'
    )
    args = parser.parse_args()

    try:
        with open(args.input_data, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input data: {e}")
        return

    total_scenes = 0
    scenes_with_agents = 0
    total_potential_pairs = 0
    total_successful_pairs = 0

    # The input data is assumed to be a list of scene dictionaries
    if not isinstance(data, list):
        print("Error: Input data must be a JSON list of scene objects.")
        return

    for scene in data:
        total_scenes += 1
        potential, successful = process_scene(scene)
        if potential > 0:
            scenes_with_agents += 1
            total_potential_pairs += potential
            total_successful_pairs += successful

    cooperation_success_rate = (
        (total_successful_pairs / total_potential_pairs) if total_potential_pairs > 0 else 0
    )

    metrics = {
        'total_scenes_processed': total_scenes,
        'scenes_with_multiple_agents': scenes_with_agents,
        'total_potential_cooperative_pairs': total_potential_pairs,
        'total_successful_cooperative_pairs': total_successful_pairs,
        'cooperation_success_rate': cooperation_success_rate,
        'cooperation_threshold_meters': COOPERATION_THRESHOLD_METERS,
        'agent_labels': AGENT_LABELS
    }

    with open(args.output_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation complete. Metrics saved to {args.output_metrics}")
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    main()
