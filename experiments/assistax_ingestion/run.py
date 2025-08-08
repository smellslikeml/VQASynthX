import argparse
import json
from pathlib import Path
import uuid

def create_vqasynth_manifest(image_dir: Path, output_file: Path):
    """
    Processes rendered images from an Assistax environment run and creates
    a manifest file compatible with the VQASynth pipeline.

    The Assistax project (https://github.com/assistive-autonomy/assistax)
    provides rich, interactive 3D environments for assistive robotics.
    This script serves as a bridge, ingesting the visual outputs from those
    simulations to generate specialized VQA data for spatial reasoning in
    robotics.

    Args:
        image_dir: Path to the directory containing rendered PNG or JPEG images
                   from an Assistax simulation.
        output_file: Path to write the output JSON manifest file.
    """
    print(f"Scanning for images in: {image_dir}")
    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    if not image_paths:
        print("No images found. Exiting.")
        return

    print(f"Found {len(image_paths)} images. Creating manifest...")

    manifest_data = []
    for img_path in image_paths:
        # Each entry in the VQASynth manifest typically needs a unique ID
        # and a path to the image. Additional metadata could be added here.
        entry = {
            "image_id": str(uuid.uuid4()),
            "image_path": str(img_path.resolve()),
            "source": "assistax_simulation"
        }
        manifest_data.append(entry)

    with open(output_file, 'w') as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Successfully created manifest at: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a VQASynth manifest from Assistax render outputs."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing rendered images from an Assistax environment."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to the output JSON manifest file."
    )
    args = parser.parse_args()
    
    create_vqasynth_manifest(args.input_dir, args.output_file)
