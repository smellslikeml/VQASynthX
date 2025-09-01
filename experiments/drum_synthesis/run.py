import os
import json
import argparse
import torch
from diffusers import DiffusionPipeline
from drum import DrUM
from tqdm import tqdm


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using DrUM for the VQASynth pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base text-to-image model ID from Hugging Face.",
    )
    parser.add_argument(
        "--tasks_file",
        type=str,
        required=True,
        help="Path to a .jsonl file with generation tasks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use.")

    args = parser.parse_args()

    # --- Environment Setup ---
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    dtype = torch.bfloat16
    device = "cuda"

    # --- Load Pipeline and Attach DrUM ---
    print(f"Loading base model: {args.model}")
    pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype).to(
        device
    )
    drum = DrUM(pipeline)
    print("DrUM attached to pipeline successfully.")

    # --- Process Generation Tasks ---
    with open(args.tasks_file, "r") as f:
        tasks = [json.loads(line) for line in f]

    print(f"Found {len(tasks)} tasks. Starting image generation...")
    for task in tqdm(tasks, desc="Generating Images"):
        prompt = task.get("prompt")
        ref = task.get("ref")
        image_id = task.get("id")
        alpha = task.get("alpha", 0.3)
        weight = task.get("weight", [1.0] * len(ref))

        if not all([prompt, ref, image_id]):
            print(f"Skipping invalid task: {task}")
            continue

        # Generate personalized image
        try:
            images = drum(prompt=prompt, ref=ref, weight=weight, alpha=alpha)

            # Save the image
            output_path = os.path.join(args.output_dir, f"{image_id}.png")
            images[0].save(output_path)

        except Exception as e:
            print(f"Failed to generate image for task {image_id}: {e}")

    print(f"Image generation complete. Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
