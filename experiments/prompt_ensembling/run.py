import torch
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

from .templates import PROMPT_TEMPLATES

def get_model_and_processor(model_name="openai/clip-vit-base-patch32"):
    """Loads a CLIP model and its processor."""
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def get_ensembled_text_features(class_name, model, processor, device):
    """Generates an ensembled text feature vector from multiple templates."""
    prompts = [template(class_name) for template in PROMPT_TEMPLATES]
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # Average the features to get the ensembled representation
    ensembled_features = text_features.mean(dim=0, keepdim=True)
    ensembled_features /= ensembled_features.norm(dim=-1, keepdim=True)
    return ensembled_features

def get_single_text_feature(class_name, model, processor, device):
    """Generates a standard single-prompt text feature vector."""
    prompt = f"a photo of a {class_name}"
    inputs = processor(text=[prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def run_localization_comparison(model, processor, image, class_name, device):
    """
    Compares localization heatmaps from single vs. ensembled prompts.
    This is a simplified simulation of a localization-by-segmentation task.
    """
    print("--- Running Localization Comparison ---")
    
    # 1. Get image features
    # To get per-patch features, we access the vision model's last hidden state
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        # Shape: (batch_size, sequence_length, hidden_size)
        image_features_patches = vision_outputs.last_hidden_state
        # We skip the CLS token
        image_features_patches = image_features_patches[:, 1:, :]


    # 2. Get text features (single and ensembled)
    print(f"Generating features for class: '{class_name}'")
    single_text_feat = get_single_text_feature(class_name, model, processor, device)
    ensembled_text_feat = get_ensembled_text_features(class_name, model, processor, device)

    # 3. Calculate similarity
    image_features_patches /= image_features_patches.norm(dim=-1, keepdim=True)
    sim_single = (image_features_patches @ single_text_feat.T).squeeze(-1)
    sim_ensembled = (image_features_patches @ ensembled_text_feat.T).squeeze(-1)
    
    print(f"Max similarity (Single Prompt):   {sim_single.max().item():.4f}")
    print(f"Max similarity (Ensembled Prompt): {sim_ensembled.max().item():.4f}")

    # 4. Reshape similarity scores into a 2D heatmap
    patch_grid_size = int(np.sqrt(sim_single.shape[1]))
    heatmap_single = sim_single.reshape(patch_grid_size, patch_grid_size).cpu().numpy()
    heatmap_ensembled = sim_ensembled.reshape(patch_grid_size, patch_grid_size).cpu().numpy()
    
    # 5. Visualize and save the results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(image)
    im1 = axes[1].imshow(heatmap_single, cmap='viridis', alpha=0.6, extent=(0, image.width, image.height, 0))
    axes[1].set_title(f"Single Prompt Heatmap (Max: {sim_single.max().item():.2f})")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1])
    
    axes[2].imshow(image)
    im2 = axes[2].imshow(heatmap_ensembled, cmap='viridis', alpha=0.6, extent=(0, image.width, image.height, 0))
    axes[2].set_title(f"Ensembled Prompt Heatmap (Max: {sim_ensembled.max().item():.2f})")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    output_path = "prompt_ensembling_comparison.png"
    plt.savefig(output_path)
    print(f"\nSaved comparison visualization to '{output_path}'")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Image from VQASynth README
    img_url = "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
    image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    class_name = "red forklift"

    print("Loading CLIP model...")
    model, processor = get_model_and_processor()
    model.to(device)

    run_localization_comparison(model, processor, image, class_name, device)
    
    print("\nExperiment finished.")
