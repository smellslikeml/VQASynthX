import torch
from diffusers import AutoPipelineForText2Image
import os

def generate_controlled_images(prompts, output_dir="data/generated_for_vqasynth"):
    """
    Generates images based on structured prompts using a diffusion model.
    This approach is inspired by Local Prompt Adaptation's principle of
    separating content/spatial information from style information within the prompt.

    Args:
        prompts (list of str): A list of detailed prompts.
        output_dir (str): Directory to save the generated images.
    """
    print("Initializing image generation pipeline...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use a powerful model like SDXL that responds well to detailed prompts.
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    print(f"Generating images for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        print(f"  - Generating image {i+1}/{len(prompts)} for prompt: '{prompt}'")
        try:
            image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            # Sanitize prompt to create a valid filename
            filename_prompt = "".join(c if c.isalnum() or c in (' ', '-') else '' for c in prompt).replace(' ', '_')
            filename = f"{i:03d}_{filename_prompt[:50]}.png"
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)
            print(f"    Saved image to {output_path}")
        except Exception as e:
            print(f"    Failed to generate image for prompt '{prompt}': {e}")

    print("Image generation complete.")

if __name__ == "__main__":
    # These prompts are structured to separate spatial relations from style,
    # inspired by the core idea of Local Prompt Adaptation.
    # This allows for generating a controlled dataset for VQASynth.
    spatial_prompts = [
        "A red cube to the left of a blue sphere, on a wooden table. Photorealistic.",
        "A green cone behind a yellow cylinder. Minimalist studio background.",
        "A small toy car on top of a large book. Cinematic lighting.",
        "A ceramic vase to the right of three apples. Still life painting style.",
        "A single fork in front of a white plate. Top-down view, high contrast.",
        "Two chairs facing each other in the middle of an empty room. Dramatic shadows.",
        "A potted plant on the far right side of a windowsill. Sunny day.",
        "A black laptop next to a white coffee mug. Clean, modern desk.",
        "A bicycle leaning against a brick wall, a backpack is on the ground to its left.",
        "An orange sits in front of a banana. Plain white background.",
    ]

    # The output of this script can serve as the input for the VQASynth pipeline.
    generate_controlled_images(spatial_prompts, output_dir="data/generated_for_vqasynth")
