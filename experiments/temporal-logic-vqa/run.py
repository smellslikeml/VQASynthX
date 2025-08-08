import json
import os
from pathlib import Path
import argparse

# This script is designed to process output from the LTLZinc generator.
# The generator creates a directory for each task, e.g., 'mnist_add'.
# Inside, it produces train/val/test splits, each with an 'images' folder
# and a 'meta.json' file.
#
# The 'meta.json' file contains the ground truth for each generated sequence:
# [
#   {
#     "trace_id": 0,
#     "trace_satisfies_formula": true,
#     "streams": {
#       "w": [ { "t": 0, "class": 1 }, { "t": 1, "class": 9 }, ... ],
#       "x": [ { "t": 0, "class": 8 }, { "t": 1, "class": 0 }, ... ],
#       "y": [ { "t": 0, "class": 4 }, { "t": 1, "class": 3 }, ... ],
#       "z": [ { "t": 0, "class": 5 }, { "t": 1, "class": 7 }, ... ]
#     }
#   },
#   ...
# ]

def load_ltlzinc_sample(dataset_path: Path, trace_id: int):
    """
    Loads a single trace (image sequence and metadata) from an LTLZinc-generated dataset.
    """
    meta_path = dataset_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"'meta.json' not found in {dataset_path}. "
                                "Please generate the LTLZinc dataset first.")

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    sample_meta = next((item for item in metadata if item['trace_id'] == trace_id), None)
    if sample_meta is None:
        raise ValueError(f"Trace with ID {trace_id} not found in metadata.")

    # Reconstruct the image sequence paths from the metadata.
    # This assumes a structure like 'images/<class_name>/<image_file>', which is typical.
    sequence_length = len(sample_meta['streams']['w'])
    image_sequence = []
    for t in range(sequence_length):
        timestep_images = {}
        for stream_name in ['w', 'x', 'y', 'z']:
            class_val = sample_meta['streams'][stream_name][t]['class']
            # We represent the image by a descriptive path placeholder.
            # A real data loader would resolve this to an actual file.
            img_path = dataset_path / "images" / str(class_val) / f"trace_{trace_id}_t_{t}_{stream_name}.png"
            timestep_images[stream_name] = str(img_path)
        image_sequence.append(timestep_images)

    return {
        "trace_id": trace_id,
        "image_sequence_paths": image_sequence,
        "satisfies_formula": sample_meta['trace_satisfies_formula']
    }

def create_temporal_vqa_prompt(sample_data):
    """
    Creates a VQA prompt for the 'mnist_add' temporal reasoning task.

    The LTL formula for this task is: "G (p(W,X,Y,Z) <-> WX !p(W,X,Y,Z))"
    where p(W,X,Y,Z) is "W + X = Y + Z".
    This translates to: At every point in time, the property p is true if and only if
    in the next step, property p is false. This defines a strict alternating pattern.
    """
    question = (
        "You will be shown a sequence of timesteps. Each timestep has four images of handwritten digits, "
        "labeled W, X, Y, and Z. "
        "Does this entire sequence satisfy the following temporal rule: 'For any given timestep, the sum of digits W+X equals Y+Z "
        "if and only if at the very next timestep, the sum W+X does NOT equal Y+Z'? "
        "Answer with only 'yes' or 'no'."
    )

    answer = "yes" if sample_data['satisfies_formula'] else "no"

    return {
        "id": f"temporal_mnist_add_{sample_data['trace_id']}",
        "image_sequence": sample_data['image_sequence_paths'],
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    }

def main():
    """
    Main function to generate and print a sample temporal VQA prompt.
    """
    parser = argparse.ArgumentParser(
        description="Generate a VQA sample for a temporal logic task from LTLZinc output."
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to a split in an LTLZinc-generated dataset (e.g., './mnist_add/test')."
    )
    parser.add_argument(
        "--trace_id",
        type=int,
        default=0,
        help="The trace ID to use for generating the sample."
    )
    args = parser.parse_args()

    print(f"Attempting to load sample from LTLZinc dataset at: {args.dataset_path}")

    try:
        sample_data = load_ltlzinc_sample(args.dataset_path, args.trace_id)
        vqa_prompt = create_temporal_vqa_prompt(sample_data)

        print("\n--- Generated Temporal VQA Sample ---")
        print(json.dumps(vqa_prompt, indent=2))
        print("\n--- End of Sample ---")

        print("\nThis demonstrates how a VQA sample can be constructed from LTLZinc output.")
        print("The 'image_sequence' contains placeholder paths to images for each timestep.")
        print("The 'conversations' key follows a standard format for VLM instruction tuning,")
        print("pairing a natural language question about a temporal rule with a ground-truth answer.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have run the LTLZinc generator and provided the correct path.")
        print("For example, using the LTLZinc Docker setup, data will be in 'benchmark/tasks/mnist_add'.")
        print(f"Then run this script with: --dataset_path /path/to/mnist_add/test")

if __name__ == "__main__":
    main()
