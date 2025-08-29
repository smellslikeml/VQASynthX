import os
import torch
import sys

# Add the project root to the Python path to allow importing 'vqasynth'
# This is necessary for running the script from the 'examples' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vqasynth.generation import generate_with_lpa_control


def main():
    """
    Runs an experiment to generate images with controlled spatial relationships and styles
    using a simplified Local Prompt Adaptation (LPA) technique.
    """
    output_dir = "output/controlled_generation"
    os.makedirs(output_dir, exist_ok=True)

    prompts = [
        {
            "name": "cube_left_sphere_photo",
            "prompt": "a red cube to the left of a blue sphere | photorealistic style",
        },
        {
            "name": "cat_on_sofa_painting",
            "prompt": "a cat sleeping on a sofa | impressionist oil painting",
        },
        {
            "name": "car_front_house_3d",
            "prompt": "a sports car parked in front of a modern house | 3d render",
        },
        {
            "name": "two_vases_watercolor",
            "prompt": "a tall vase behind a short vase | watercolor sketch",
        },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for item in prompts:
        print(f"\nGenerating image for prompt: '{item['prompt']}'")
        try:
            image = generate_with_lpa_control(
                prompt=item["prompt"],
                device=device,
                num_inference_steps=30,
                guidance_scale=8.0,
                style_control_end_step=0.75,
            )
            output_path = os.path.join(output_dir, f"{item['name']}.png")
            image.save(output_path)
            print(f"Saved image to {output_path}")
        except Exception as e:
            print(f"Failed to generate image for prompt '{item['prompt']}': {e}")


if __name__ == "__main__":
    main()
