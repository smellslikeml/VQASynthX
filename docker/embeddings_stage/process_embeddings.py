import argparse
import json
import os
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


def generate_semantic_embedding(model, processor, image_path, device):
    """Generates a semantic embedding for a given image."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features.squeeze(0).cpu()
    except Exception as e:
        print(f"Warning: Could not process image {image_path}. Error: {e}")
        return None


def create_positional_embedding(bbox, img_width, img_height):
    """
    Creates a normalized positional embedding from a bounding box.
    The embedding is [x_min, y_min, x_max, y_max], all normalized to [0, 1].
    """
    x1, y1, x2, y2 = bbox
    normalized_bbox = [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height]
    return torch.tensor(normalized_bbox, dtype=torch.float32)


def main():
    """
    Processes object detections to generate multi-view embeddings.
    Inspired by the multi-view featurization strategy from the DataCentricBrainGraphs paper,
    this script combines semantic embeddings (from CLIP) with structural embeddings
    (normalized bounding box coordinates) to create a richer object representation.
    """
    parser = argparse.ArgumentParser(
        description="Generate multi-view object embeddings."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input JSONL file with object detections.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSONL file with embeddings.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing the source images.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a pre-trained model for semantic embeddings
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_path, "r") as infile, open(args.output_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            image_path = os.path.join(args.image_dir, data["image_filename"])

            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}. Skipping.")
                continue

            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(
                    f"Warning: Could not open image {image_path}. Error: {e}. Skipping."
                )
                continue

            # For this experiment, we generate embeddings for cropped objects.
            base_image = Image.open(image_path).convert("RGB")

            processed_objects = []
            for obj in data.get("objects", []):
                bbox = obj.get("bounding_box")  # Expected format: [x1, y1, x2, y2]
                if not bbox or len(bbox) != 4:
                    print(
                        f"Warning: Skipping object with invalid bounding box in {data['image_filename']}"
                    )
                    continue

                # View 1: Semantic Embedding from the cropped object
                cropped_image = base_image.crop(bbox)
                inputs = processor(images=cropped_image, return_tensors="pt").to(device)
                with torch.no_grad():
                    semantic_embedding = (
                        model.get_image_features(**inputs).squeeze(0).cpu()
                    )

                # View 2: Structural/Positional Embedding
                positional_embedding = create_positional_embedding(
                    bbox, img_width, img_height
                )

                # Concatenate the two views to create the multi-view embedding
                multi_view_embedding = torch.cat(
                    (semantic_embedding, positional_embedding), dim=0
                )

                obj["multi_view_embedding"] = multi_view_embedding.tolist()
                obj["semantic_embedding_dim"] = len(semantic_embedding)
                obj["positional_embedding_dim"] = len(positional_embedding)
                processed_objects.append(obj)

            data["objects"] = processed_objects
            outfile.write(json.dumps(data) + "\n")

    print(f"Processing complete. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()
