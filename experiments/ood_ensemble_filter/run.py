import torch
import open_clip
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os
import json
import requests

# This script is a minimal implementation of the COOkeD ensemble idea for OOD detection.
# It combines a zero-shot CLIP model with a standard ResNet classifier to filter
# images unsuitable for the VQASynth spatial reasoning pipeline.
# Dependencies: pip install torch torchvision open-clip-torch Pillow requests

def get_imagenet_classes():
    """Downloads and returns the ImageNet-1k class mapping."""
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        class_idx = response.json()
        # The dictionary is { 'class_index': ['class_id', 'class_name'] }
        # We just want a list of class names in the correct order.
        return [class_idx[str(k)][1] for k in range(len(class_idx))]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ImageNet classes: {e}")
        return None

def main():
    """Main function to run the OOD detection experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Models (inspired by COOkeD's heterogeneous ensemble) ---
    # a) Zero-shot CLIP model
    print("Loading CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained='laion400m_e32', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')
    clip_model.eval()

    # b) Standard classifier (ResNet18 trained on ImageNet)
    print("Loading ResNet18 model...")
    classifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    classifier.eval()

    # --- 2. Define In-Distribution Classes and Prompts ---
    print("Fetching ImageNet classes...")
    imagenet_classes = get_imagenet_classes()
    if not imagenet_classes:
        return
    
    # Create text prompts for CLIP
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in imagenet_classes]
    with torch.no_grad():
        text_features = tokenizer(prompts).to(device)
        text_features = clip_model.encode_text(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # --- 3. Define Preprocessing and OOD Scoring ---
    # NOTE: The VQASynth pipeline would use its own normalization. For this OOD filter,
    # we need separate transforms for each model, as highlighted in the COOkeD paper.
    classifier_preprocess = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

    # OOD scoring function: Maximum Softmax Probability (MSP)
    ood_scoring = lambda softmax_probs: torch.max(softmax_probs, dim=1).values.item()

    # --- 4. Process Images and Calculate OOD Scores ---
    image_dir = "test_images"
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f'\nError: Please create a directory named "{image_dir}" and add test images to it.')
        return

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_path in image_paths:
        print(f"\n{'='*20} {os.path.basename(image_path)} {'='*20}")
        image = Image.open(image_path).convert("RGB")

        # Preprocess image for each model
        image_clip = clip_preprocess(image).unsqueeze(0).to(device)
        image_classifier = classifier_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # a) Get zero-shot CLIP prediction
            image_features = clip_model.encode_image(image_clip)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sim = (100.0 * image_features @ text_features.T)
            softmax_clip = sim.softmax(dim=-1)

            # b) Get standard classifier prediction
            logits_classifier = classifier(image_classifier)
            softmax_classifier = logits_classifier.softmax(dim=-1)

        # c) Combined prediction (the COOkeD ensemble)
        # For simplicity, we average the softmax probabilities.
        softmax_ensemble = torch.stack([softmax_clip, softmax_classifier]).mean(0)

        # --- 5. Report Predictions and OOD Scores ---
        pred_clip = softmax_clip.argmax(dim=1).item()
        pred_classifier = softmax_classifier.argmax(dim=1).item()
        pred_ensemble = softmax_ensemble.argmax(dim=1).item()

        msp_clip = ood_scoring(softmax_clip)
        msp_classifier = ood_scoring(softmax_classifier)
        msp_ensemble = ood_scoring(softmax_ensemble)

        print(f"CLIP Prediction: {imagenet_classes[pred_clip]} (MSP: {msp_clip:.3f})")
        print(f"ResNet Prediction: {imagenet_classes[pred_classifier]} (MSP: {msp_classifier:.3f})")
        print(f"--> COOkeD Ensemble Prediction: {imagenet_classes[pred_ensemble]} (MSP: {msp_ensemble:.3f})")

        # A simple threshold could be used to filter images.
        # e.g., if msp_ensemble > 0.6, consider it In-Distribution for VQASynth.
        threshold = 0.6
        verdict = "In-Distribution (Suitable)" if msp_ensemble > threshold else "Out-of-Distribution (Unsuitable)"
        print(f"\nVerdict (Threshold={threshold}): {verdict}")

if __name__ == "__main__":
    main()
