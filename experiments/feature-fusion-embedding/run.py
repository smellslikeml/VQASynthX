import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import timm
import requests
import numpy as np

# This script demonstrates the core idea from the MedARC/algonauts2025 repository:
# fusing features from multiple diverse models to create a richer representation.
#
# SOURCE EVIDENCE: The `generate_configs.py` file in the source repo systematically
# combines features from models like Llama, Whisper, Qwen, InternVL, and VJEPA.
# This experiment adapts that "feature ensemble" strategy for the VQASynth context.
#
# Here, we fuse features from a vision-language model (CLIP) and a self-supervised
# vision model (DINOv2) to create a composite embedding for an image.

def get_device():
    """Gets the available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_models(device):
    """Loads a set of diverse vision models."""
    models = {}
    
    # Model 1: CLIP (Vision-Language)
    # Represents semantic understanding trained on image-text pairs.
    clip_id = "openai/clip-vit-base-patch32"
    models['clip'] = {
        'model': AutoModel.from_pretrained(clip_id).to(device).vision_model,
        'processor': AutoProcessor.from_pretrained(clip_id)
    }
    
    # Model 2: DINOv2 (Self-Supervised Vision)
    # Represents structural and textural features learned without labels.
    # Analogous to using VJEPA in the source repository.
    dino_id = "facebook/dinov2-base"
    models['dino'] = {
        'model': AutoModel.from_pretrained(dino_id).to(device),
        'processor': AutoProcessor.from_pretrained(dino_id)
    }

    print("Loaded models: CLIP (ViT-B/32), DINOv2 (ViT-B/14)")
    return models

def get_fused_embedding(image, models, device):
    """
    Extracts and concatenates embeddings from multiple models for a given image.
    """
    all_features = []
    
    # --- CLIP Feature Extraction ---
    clip_model_info = models['clip']
    clip_inputs = clip_model_info['processor'](images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_outputs = clip_model_info['model'](**clip_inputs)
        # We use the pooled output which represents the entire image [batch_size, embed_dim]
        clip_features = clip_outputs.pooler_output
    all_features.append(clip_features)
    print(f"Extracted CLIP features with shape: {clip_features.shape}")

    # --- DINOv2 Feature Extraction ---
    dino_model_info = models['dino']
    dino_inputs = dino_model_info['processor'](images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        dino_outputs = dino_model_info['model'](**dino_inputs)
        # DINOv2 provides a pooler_output as well, summarizing the [CLS] token
        dino_features = dino_outputs.pooler_output
    all_features.append(dino_features)
    print(f"Extracted DINOv2 features with shape: {dino_features.shape}")

    # --- Feature Fusion ---
    # The simplest fusion strategy is concatenation, as used in many ensemble methods.
    fused_embedding = torch.cat(all_features, dim=1)
    print(f"Concatenated features to create fused embedding.")
    
    return fused_embedding

if __name__ == "__main__":
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Load a sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print(f"Loaded sample image from {url}")
    except Exception as e:
        print(f"Could not load sample image. Creating a dummy image. Error: {e}")
        image = Image.fromarray(np.uint8(np.random.rand(480, 640, 3) * 255))


    # Load models
    models = load_models(device)
    
    # Generate fused embedding
    fused_embedding = get_fused_embedding(image, models, device)
    
    # Print results
    print("\n--- EXPERIMENT RESULTS ---")
    print(f"Final fused embedding shape: {fused_embedding.shape}")
    print("This demonstrates that features from multiple distinct models can be successfully combined into a single tensor.")
    print("This fused representation can now be used in downstream tasks within VQASynth, potentially improving the quality of generated VQA data.")
