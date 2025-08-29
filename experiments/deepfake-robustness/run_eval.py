import torch
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from PIL import Image
import requests
import os

# --- Configuration ---
MODEL_ID = "llava-hf/llava-next-vicuna-7b-hf"
# Sample images from the SOURCE repo's associated Hugging Face space
# Real image: https://huggingface.co/spaces/Sumsub/Deepfake-Game/blob/main/examples/00000.png
REAL_IMAGE_URL = (
    "https://huggingface.co/spaces/Sumsub/Deepfake-Game/resolve/main/examples/00000.png"
)
# Deepfake image: https://huggingface.co/spaces/Sumsub/Deepfake-Game/blob/main/examples/00001.png
FAKE_IMAGE_URL = (
    "https://huggingface.co/spaces/Sumsub/Deepfake-Game/resolve/main/examples/00001.png"
)
OUTPUT_DIR = "evaluation_outputs"


# --- VLM Setup ---
def get_model_and_processor():
    """Loads the LLaVA-NeXT model and processor with 4-bit quantization."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, processor


# --- Image Handling ---
def download_image(url, filename):
    """Downloads an image from a URL and saves it."""
    if os.path.exists(filename):
        print(f"Image already exists: {filename}")
        return Image.open(filename).convert("RGB")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        image.save(filename)
        print(f"Downloaded and saved {filename}")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None


# --- Evaluation Logic ---
def run_evaluation(model, processor, image, image_type):
    """Runs a VQA query on a single image and prints the result."""
    # VQASynth focuses on spatial reasoning. This prompt probes perception of the person and their environment.
    # An ideal robust model might comment on unnatural features in the fake image.
    prompt = "[INST] <image>\nDescribe the person in this image and their immediate surroundings. Does anything look unusual? [/INST]"

    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)

    print(f"\n--- Running Evaluation on {image_type.upper()} Image ---")
    print(f"Prompt: {prompt.replace('<image>', '').strip()}")

    # Generate response
    output = model.generate(**inputs, max_new_tokens=200)
    response = processor.decode(output[0], skip_special_tokens=True)

    # Clean up the response to only show the assistant's part
    assistant_response = response.split("[/INST]")[-1].strip()
    print(f"Model Response:\n{assistant_response}")
    print("--------------------------------------------------")


# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing deepfake robustness evaluation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model
    print(f"Loading model: {MODEL_ID}")
    model, processor = get_model_and_processor()

    # 2. Download Images
    print("Downloading test images...")
    real_image_path = os.path.join(OUTPUT_DIR, "real_image.png")
    fake_image_path = os.path.join(OUTPUT_DIR, "deepfake_image.png")
    real_image = download_image(REAL_IMAGE_URL, real_image_path)
    fake_image = download_image(FAKE_IMAGE_URL, fake_image_path)

    # 3. Run Evaluations
    if real_image:
        run_evaluation(model, processor, real_image, "real")
    if fake_image:
        run_evaluation(model, processor, fake_image, "deepfake")

    print("\nEvaluation complete.")
