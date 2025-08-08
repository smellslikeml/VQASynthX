import argparse
import json
from datasets import load_dataset
import torch

# NOTE: This script requires `torch_geometric` to be installed.
# pip install torch_geometric
from torch_geometric.data import Data

def get_object_properties(obj):
    """Extracts bounding box center and dimensions from a VQASynth object entry."""
    # Assumes box_2d is [xmin, ymin, xmax, ymax]
    bbox = obj.get('box_2d', [0, 0, 0, 0])
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    cx, cy = x_min + w / 2, y_min + h / 2
    return {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'name': obj.get('label', 'unknown')}

def is_left_of(props1, props2):
    """Check if the center of object 1 is to the left of the center of object 2."""
    return props1['cx'] < props2['cx']

def is_taller_than(props1, props2):
    """Check if object 1 is taller than object 2 based on bbox height."""
    return props1['h'] > props2['h']

def parse_question_and_objects(question, objects):
    """Rudimentary parser to find the relation and the two objects being compared."""
    q_lower = question.lower()
    # This is a major simplification. It assumes the question mentions two objects
    # that are present and first in the `objects` list.
    # A real version needs robust entity linking.
    if len(objects) < 2:
        return None, None, None
    
    obj1, obj2 = objects[0], objects[1]

    if 'left side of' in q_lower or 'left of' in q_lower:
        return 'left_of', obj1, obj2
    if 'greater height compared to' in q_lower or 'taller than' in q_lower:
        return 'taller_than', obj1, obj2
    
    return None, None, None

def evaluate_sample(sample):
    """
    Evaluates a single VQA sample by building and analyzing a scene graph.
    """
    try:
        # VQASynth datasets often store rich data in a 'metadata' JSON string
        # or directly in conversation turns.
        if 'metadata' in sample and isinstance(sample['metadata'], str):
            metadata = json.loads(sample['metadata'])
        else:
            # Fallback for other formats
            metadata = sample

        objects = metadata.get('objects', [])
        question = sample.get('question')
        model_answer = sample.get('answer')

        if not all([objects, question, model_answer]):
            return "skipped_missing_data"

    except (json.JSONDecodeError, KeyError, TypeError):
        return "skipped_parsing_error"

    relation, obj1_data, obj2_data = parse_question_and_objects(question, objects)

    if not relation:
        return "skipped_unsupported_question"

    # 1. Represent scene with PyG (nodes only for this experiment)
    obj1_props = get_object_properties(obj1_data)
    obj2_props = get_object_properties(obj2_data)

    node_features = torch.tensor([
        [obj1_props['cx'], obj1_props['cy'], obj1_props['w'], obj1_props['h']],
        [obj2_props['cx'], obj2_props['cy'], obj2_props['w'], obj2_props['h']]
    ], dtype=torch.float)
    
    # A PyG Data object, though we only use the features directly here
    _ = Data(x=node_features)

    # 2. Determine Ground Truth from geometric properties
    ground_truth_bool = False
    if relation == 'left_of':
        ground_truth_bool = is_left_of(obj1_props, obj2_props)
    elif relation == 'taller_than':
        ground_truth_bool = is_taller_than(obj1_props, obj2_props)

    # 3. Parse model's boolean answer
    # VQASynth answers are often structured, e.g., "<think>...</think> <answer>Correct..."
    answer_text = model_answer.split('<answer>')[-1].strip().lower()
    
    # Simple keyword-based check for boolean sentiment
    positive_indicators = ['correct', 'indeed', 'yes']
    negative_indicators = ['incorrect', 'not', 'no']

    model_answer_bool = None
    if any(indicator in answer_text for indicator in positive_indicators):
        model_answer_bool = True
    elif any(indicator in answer_text for indicator in negative_indicators):
        model_answer_bool = False

    if model_answer_bool is None:
        return "skipped_ambiguous_answer"

    return ground_truth_bool == model_answer_bool

def main(args):
    """Main evaluation loop."""
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split='train')

    correct_count = 0
    total_evaluated = 0
    skipped_counts = {}

    for i, sample in enumerate(dataset):
        if args.num_samples is not None and i >= args.num_samples:
            break
        
        result = evaluate_sample(sample)

        if isinstance(result, bool):
            total_evaluated += 1
            if result:
                correct_count += 1
        else:
            skipped_counts[result] = skipped_counts.get(result, 0) + 1

    print("\n--- Evaluation Finished ---")
    if total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        print(f"Total Samples Evaluated: {total_evaluated}")
        print(f"Correct Predictions: {correct_count}")
        print(f"Geometric Consistency Score: {accuracy:.2f}%")
    else:
        print("No samples could be evaluated with the current logic.")
    
    print("\nSkipped Sample Reasons:")
    for reason, count in sorted(skipped_counts.items()):
        print(f"- {reason}: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate VQA datasets using scene graph logic.")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='remyxai/OpenSpaces_MC_R1',
        help='Name of the Hugging Face dataset to evaluate.'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to process from the dataset. Default is 100.'
    )
    args = parser.parse_args()
    main(args)
