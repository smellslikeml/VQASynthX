import json
import random
import argparse
from pathlib import Path

# Assuming the vqasynth package is installed in the Docker container
from vqasynth.prompts.templates import SPATIAL_RELATIONSHIP_TEMPLATES


def load_scene_data(input_path):
    """Loads scene data from a JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def generate_qa_pairs(scene_data):
    """Generates VQA pairs from scene data using templates."""
    qa_pairs = []
    # Example input format from a previous stage:
    # {
    #   "image_id": "img_001.jpg",
    #   "relationships": [
    #     {"obj1": "red forklift", "obj2": "brown boxes", "relation": "left_of", "answer": "Yes"}
    #   ]
    # }
    for relationship in scene_data.get("relationships", []):
        obj1 = relationship.get("obj1")
        obj2 = relationship.get("obj2")
        relation = relationship.get("relation")
        answer = relationship.get("answer")

        if not all([obj1, obj2, relation, answer]):
            continue

        # Get the appropriate templates for the relation
        if relation in SPATIAL_RELATIONSHIP_TEMPLATES:
            template_functions = SPATIAL_RELATIONSHIP_TEMPLATES[relation]
            # Pick one template randomly to generate the question
            template = random.choice(template_functions)
            question = template(obj1, obj2)

            qa_pairs.append(
                {
                    "id": f"{scene_data['image_id']}_{len(qa_pairs)}",
                    "image": scene_data["image_id"],
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )
    return qa_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate VQA prompts from scene data using templates."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing scene data JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated VQA pairs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_qa_pairs = []
    for json_file in sorted(list(input_path.glob("*.json"))):
        scene_data = load_scene_data(json_file)
        qa_pairs = generate_qa_pairs(scene_data)
        all_qa_pairs.extend(qa_pairs)
        print(f"Processed {json_file.name}, generated {len(qa_pairs)} QA pairs.")

    if all_qa_pairs:
        output_file = output_path / "final_prompt_dataset.json"
        with open(output_file, "w") as f:
            json.dump(all_qa_pairs, f, indent=2)
        print(f"Wrote {len(all_qa_pairs)} total QA pairs to {output_file}")


if __name__ == "__main__":
    main()
