import argparse
import os
from pathlib import Path

import imageio
import torch
from diffusers import AutoPipelineForImage2Video
from diffusers.utils import export_to_video
from PIL import Image


def generate_dynamic_scene(
    image_path: str,
    prompt: str,
    output_path: str,
    model_id: str = "stdstu123/Yume-I2V-540P",
    device: str = "cuda",
):
    """
    Generates a video from a static image and a text prompt using the Yume model.

    Args:
        image_path (str): Path to the input static image.
        prompt (str): Text prompt describing the desired motion.
        output_path (str): Path to save the generated MP4 video.
        model_id (str): The Hugging Face model ID for the Yume model.
        device (str): The device to run the model on ('cuda' or 'cpu').
    """
    print(f"Loading model: {model_id}")
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, switching to CPU. This will be very slow.")
        device = "cpu"

    # Although Yume has a complex codebase, its core model is compatible with diffusers.
    # We use AutoPipelineForImage2Video for a minimal, standard integration.
    pipeline = AutoPipelineForImage2Video.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    print(f"Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input image not found at {image_path}")
        return

    print(f"Generating video with prompt: '{prompt}'")
    video_frames = pipeline(prompt=prompt, image=image, num_inference_steps=25, num_frames=24).frames[0]

    # Ensure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting video to: {output_path}")
    export_to_video(video_frames, output_path, fps=8)
    print("Video generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dynamic scene video from a static image using the Yume model."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input static image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text prompt describing the motion. Example: "This video depicts a city walk scene..."',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output/generated_scene.mp4",
        help="Path to save the output mp4 video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )

    args = parser.parse_args()

    generate_dynamic_scene(
        image_path=args.image_path,
        prompt=args.prompt,
        output_path=args.output_path,
        device=args.device,
    )
