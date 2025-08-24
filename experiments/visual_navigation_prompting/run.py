import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import requests
import os


def generate_navigation_instructions(
    model, processor, start_image_path, goal_image_path
):
    """
    Generates navigation instructions from a start and goal image using a VLM.

    Args:
        model: The loaded VLM model.
        processor: The VLM processor.
        start_image_path (str): Path to the starting view image.
        goal_image_path (str): Path to the goal view image.

    Returns:
        str: The generated navigation instructions.
    """
    try:
        start_image = Image.open(start_image_path).convert("RGB")
        goal_image = Image.open(goal_image_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Error: Could not find image file. {e}")
        print(
            "Please ensure you have downloaded the sample images into the 'assets' directory."
        )
        return None

    # This prompt is inspired by the GoViG paper's task definition:
    # "generating navigation instructions directly from egocentric visual observations
    # of the initial and goal states."
    prompt = (
        "USER: <image>\n<image>\nYou are an embodied AI agent. Your task is to generate navigation instructions. "
        "The first image shows your initial view. The second image shows your goal view. "
        "Based on these two images, provide a sequence of clear, step-by-step instructions to navigate from the start to the goal.\nASSISTANT:"
    )

    inputs = processor(
        text=prompt, images=[start_image, goal_image], return_tensors="pt"
    ).to(model.device)

    generate_ids = model.generate(**inputs, max_new_tokens=200)

    # The output needs to be decoded, skipping special tokens and the prompt.
    # The length of the prompt tokens needs to be subtracted from the generated IDs.
    input_token_len = inputs.input_ids.shape[1]
    decoded_text = processor.batch_decode(
        generate_ids[:, input_token_len:], skip_special_tokens=True
    )[0]

    return decoded_text.strip()


def download_image(url, save_path):
    """Downloads an image from a URL and saves it."""
    if not os.path.exists(save_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {os.path.basename(save_path)}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return False
    return True


def main():
    """
    Main function to run the visual navigation instruction generation experiment.
    """
    print("Initializing model and processor...")
    # Using a well-known VLM suitable for this task.
    model_id = "llava-hf/llava-1.5-7b-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Model and processor loaded on {device}.")

    # Setup asset paths
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)

    # Example images from the GoViG repository for demonstration
    start_image_url = "https://raw.githubusercontent.com/F1y1113/GoViG/main/assists/seen/example1/onepass/initial_obs_0.png"
    goal_image_url = "https://raw.githubusercontent.com/F1y1113/GoViG/main/assists/seen/example1/onepass/goal_obs.png"

    start_image_filename = "start_view.png"
    goal_image_filename = "goal_view.png"

    start_image_path = os.path.join(assets_dir, start_image_filename)
    goal_image_path = os.path.join(assets_dir, goal_image_filename)

    # Download images if they don't exist
    print("Checking for sample images...")
    if not download_image(start_image_url, start_image_path) or not download_image(
        goal_image_url, goal_image_path
    ):
        print("Failed to download sample images. Exiting.")
        return

    print("\n--- Generating Navigation Instructions ---")
    print(f"Start Image: {start_image_path}")
    print(f"Goal Image:  {goal_image_path}")

    instructions = generate_navigation_instructions(
        model, processor, start_image_path, goal_image_path
    )

    if instructions:
        print("\nGenerated Instructions:")
        print("-------------------------")
        print(instructions)
        print("-------------------------")


if __name__ == "__main__":
    main()
