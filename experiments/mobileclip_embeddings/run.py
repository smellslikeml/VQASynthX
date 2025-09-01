import torch
import open_clip
from PIL import Image
import requests
import os


def download_image(url, save_path):
    """Downloads an image from a URL and saves it locally."""
    if not os.path.exists(save_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded image to {save_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            return False
    return True


def main():
    """
    This script demonstrates the integration of MobileCLIP for generating
    image and text embeddings within the VQASynth experimental framework.
    It uses the efficient MobileCLIP-S0 model via the open_clip library.
    """
    print("--- MobileCLIP Embedding Generation Experiment ---")

    # Use a sample image from the VQASynth README
    image_url = "https://github.com/smellslikeml/experimental-vqasynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
    image_path = "warehouse_sample_1.jpeg"

    if not download_image(image_url, image_path):
        print("Exiting due to image download failure.")
        return

    # Check for CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load MobileCLIP model and preprocessor
    # Following the official usage example from apple/ml-mobileclip README
    # Using pretrained='datacompdr' will automatically download from HuggingFace Hub.
    try:
        print("Loading MobileCLIP-S0 model...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S0", pretrained="datacompdr", device=device
        )
        tokenizer = open_clip.get_tokenizer("MobileCLIP-S0")
        print("Model loaded successfully.")
    except Exception as e:
        print(
            f"Failed to load model. Ensure open_clip is installed correctly. Error: {e}"
        )
        print("Try: pip install open_clip_torch")
        return

    # Prepare image and text inputs
    try:
        image = (
            preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        )
        # Text prompts relevant to the VQASynth task of spatial reasoning
        text_prompts = [
            "A red forklift on the left side of brown cardboard boxes",
            "A man in a red hat walking near a wooden pallet",
            "A warehouse with concrete floors",
            "A sunny day at the beach",
        ]
        text = tokenizer(text_prompts).to(device)
        print(f"Processing image: {image_path}")
        print(f"Processing text prompts: {text_prompts}")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return

    # Generate embeddings and calculate similarity
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity probabilities
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("\n--- Results ---")
    print("Similarity probabilities between image and text prompts:")
    for prompt, prob in zip(text_prompts, text_probs[0]):
        print(f"- '{prompt}': {prob.item():.4f}")

    print("\nExperiment finished successfully.")
    print("This demonstrates that MobileCLIP can be integrated to generate embeddings,")
    print("a foundational step for the VQASynth data generation pipeline.")


if __name__ == "__main__":
    main()
