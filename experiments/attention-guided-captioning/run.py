#!/usr/bin/env python
#
# This script implements a self-contained experiment for attention-guided image captioning.
# The core logic is adapted from the AGIC project.
# Source: https://github.com/saitejalekkala33/AGIC-code
# Evidence File: AGIC/agic_ablation.py

import torch
import torchvision.transforms as T
from PIL import Image
import requests
import argparse
import logging
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_attention_map(
    model: Blip2ForConditionalGeneration, image_tensor: torch.Tensor, image_size: tuple
) -> torch.Tensor:
    """
    Extracts the mean attention map from the vision model's final layer.
    """
    logger.info("Extracting attention map from vision model...")
    model.vision_model.config.output_attentions = True

    with torch.no_grad():
        outputs = model.vision_model(pixel_values=image_tensor, output_attentions=True)

    # Use the attention from the last transformer block
    attentions = outputs.attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)

    # Average attention across heads for the [CLS] token's attention on other patches
    # Shape: (num_heads, seq_len-1) -> mean -> (seq_len-1)
    attn_map = attentions[0, :, 0, 1:].mean(0)

    # Reshape to a grid (e.g., 16x16 for 224x224 input) and normalize
    patch_grid_size = int((attn_map.shape[0]) ** 0.5)
    attn_map = attn_map.reshape(patch_grid_size, patch_grid_size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    # Resize to original image dimensions for amplification
    attn_map_resized = T.functional.resize(
        attn_map.unsqueeze(0),
        [image_size[0], image_size[1]],
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    )[0]

    return attn_map_resized


def amplify_image_with_attention(
    image_tensor: torch.Tensor, attn_map: torch.Tensor, k: float
) -> torch.Tensor:
    """
    Amplifies the image tensor using the attention map and amplification factor k.
    Formula: new_pixel = old_pixel * (1 + k * attention_value)
    """
    logger.info(f"Amplifying image with factor k={k}...")
    attn_map = attn_map.to(image_tensor.device).unsqueeze(0)  # Add channel dim
    amplified_tensor = image_tensor * (1 + k * attn_map)

    # Clamp values to the valid range for the normalized tensor (BLIP processor uses [0, 1])
    return torch.clamp(amplified_tensor, 0.0, 1.0)


def generate_caption(
    model: Blip2ForConditionalGeneration,
    processor: AutoProcessor,
    pixel_values: torch.Tensor,
    device: torch.device,
    prompt: str = "a photo of",
) -> str:
    """Generates a caption for a given image tensor."""
    # Create inputs dictionary, but use the provided pixel_values
    inputs = processor(text=prompt, return_tensors="pt").to(device, torch.float16)
    inputs["pixel_values"] = pixel_values

    generated_ids = model.generate(**inputs, max_new_tokens=30)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    return generated_text


def main(args):
    """Main function to run the attention-guided captioning experiment."""

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_id = "Salesforce/blip2-opt-2.7b"
    logger.info(f"Loading model: {model_id}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # --- Load and Process Image ---
    logger.info(f"Loading image from URL: {args.image_url}")
    try:
        raw_image = Image.open(requests.get(args.image_url, stream=True).raw).convert(
            "RGB"
        )
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    image_size = raw_image.size
    inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    # --- 1. Baseline Caption Generation ---
    logger.info("=" * 30)
    logger.info("STEP 1: Generating baseline caption...")
    baseline_caption = generate_caption(model, processor, pixel_values, device)
    print(f"\n[BASELINE CAPTION]:\n{baseline_caption}\n")

    # --- 2. Attention-Guided Caption Generation ---
    logger.info("=" * 30)
    logger.info("STEP 2: Generating attention-guided caption...")

    # Extract attention map from the original image tensor
    attention_map = extract_attention_map(model, pixel_values, image_size)

    # Amplify the original image tensor using the attention map
    amplified_pixel_values = amplify_image_with_attention(
        pixel_values, attention_map, k=args.k
    )

    # Generate caption from the amplified tensor
    guided_caption = generate_caption(model, processor, amplified_pixel_values, device)

    print(f"[ATTENTION-GUIDED CAPTION (k={args.k})]:\n{guided_caption}\n")
    logger.info("=" * 30)
    logger.info("Experiment finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Attention-Guided Image Captioning Experiment."
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://raw.githubusercontent.com/smellslikeml/experimental-vqasynth/main/assets/warehouse_sample_1.jpeg",
        help="URL of the image to process.",
    )
    parser.add_argument(
        "-k",
        type=float,
        default=5.0,
        help="Amplification factor for the attention map.",
    )
    args = parser.parse_args()
    main(args)
