import torch
from diffusers import ImageToVideoSDPipeline
from diffusers.utils import export_to_video, load_image
import requests
from PIL import Image
from io import BytesIO

def generate_yume_trajectory(image_url: str, prompt: str, output_path: str = "yume_trajectory.mp4"):
    """
    Generates a video trajectory from a starting image and an action-based prompt
    using the YUME model.

    Args:
        image_url (str): URL of the starting image.
        prompt (str): A detailed, structured prompt describing the action.
        output_path (str): Path to save the generated video.
    """
    # The YUME README specifies a model available on Hugging Face.
    # We use a standard ImageToVideo pipeline as a proxy for this experiment.
    # The exact pipeline might need adjustment based on YUME's specific architecture,
    # but this provides a minimal, testable starting point.
    model_id = "stdstu123/Yume-I2V-540P"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        pipe = ImageToVideoSDPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
        pipe.to(device)
    except Exception as e:
        print(f"Could not load the pipeline from Hub with default class: {e}")
        print("This may be because YUME requires a custom pipeline class.")
        print("This script serves as a template for the experimental logic.")
        # As a fallback, create dummy frames to allow the script to complete.
        dummy_frame = load_image(image_url).resize((512, 512))
        video_frames = [dummy_frame for _ in range(16)]
        export_to_video(video_frames, output_path, fps=7)
        print(f"Generated a dummy video at {output_path} as the model could not be loaded.")
        return

    print(f"Loading initial image from {image_url}...")
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")
        return

    print(f"Generating video with prompt: '{prompt}'")
    video_frames = pipe(prompt=prompt, image=init_image, num_inference_steps=25, num_frames=16).frames
    
    print(f"Exporting video to {output_path}...")
    export_to_video(video_frames, output_path, fps=7)
    print("Generation complete.")


if __name__ == "__main__":
    # Use a sample image from VQASynth's README, relevant to its domain.
    sample_image_url = "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
    
    # Use a structured prompt inspired by YUME's caption.txt to simulate an action.
    action_prompt = (
        "This video depicts a warehouse scene with a first-person view (FPV)."
        "Person moves forward (W).Camera remains still (·)."
        "Actual distance moved:5.0 at 100 meters per second."
        "Angular change rate (turn speed):0.0."
        "View rotation speed:0.0."
    )

    generate_yume_trajectory(image_url=sample_image_url, prompt=action_prompt)
