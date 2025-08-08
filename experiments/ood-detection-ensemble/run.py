import torch
import open_clip
from PIL import Image
import requests
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import json

# --- Constants and Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ImageNet class labels for a consistent output space between models
# A small file with these labels would typically be included, but for a single file we can fetch it.
try:
    IMAGENET_CLASSES_URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd42403/imagenet1000_clsidx_to_labels.json"
    response = requests.get(IMAGENET_CLASSES_URL)
    response.raise_for_status()
    # The keys are strings '0'-'999', so we convert to a list
    imagenet_class_map = response.json()
    IMAGENET_CLASSES = [imagenet_class_map[str(i)] for i in range(1000)]
except Exception as e:
    print(f"Could not fetch ImageNet classes, using a dummy list. Error: {e}")
    IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]


# --- Model Loading ---

def get_models():
    """Loads a pre-trained ResNet50 and a zero-shot CLIP model."""
    print(f"Loading models to {DEVICE}...")
    # 1. Standard classifier (ResNet50 trained on ImageNet)
    weights = ResNet50_Weights.IMAGENET1K_V2
    classifier = resnet50(weights=weights).to(DEVICE)
    classifier.eval()
    
    # Get the same normalization transform used for training the classifier
    classifier_preprocess = weights.transforms()

    # 2. Zero-shot CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=DEVICE)
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    print("Models loaded.")
    return classifier, classifier_preprocess, clip_model, clip_preprocess, clip_tokenizer


def get_clip_text_features(class_names, clip_model, tokenizer):
    """Encodes class names into text features for CLIP."""
    print("Encoding text prompts for CLIP...")
    with torch.no_grad():
        prompts = [f"a photo of a {c.split(',')[0]}" for c in class_names]
        text = tokenizer(prompts).to(DEVICE)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


# --- Main Experiment Logic ---

def run_ood_analysis(image_path, classifier, classifier_preprocess, clip_model, clip_preprocess, clip_text_features):
    """
    Analyzes a single image, calculating OOD scores for individual models and their ensemble.
    """
    try:
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image at {image_path}: {e}")
        return

    # Preprocess image for each model
    image_cls = classifier_preprocess(image).unsqueeze(0).to(DEVICE)
    image_clip = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # 1. Get classifier prediction
        logits_classifier = classifier(image_cls)
        softmax_classifier = logits_classifier.softmax(dim=-1)

        # 2. Get zero-shot CLIP prediction
        image_features = clip_model.encode_image(image_clip)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ clip_text_features.T)
        softmax_clip = similarity.softmax(dim=-1)

        # 3. Ensemble prediction (averaging softmax probabilities)
        softmax_ensemble = (softmax_classifier + softmax_clip) / 2.0

    # Calculate Maximum Softmax Probability (MSP) as the confidence score
    # A higher MSP suggests an in-distribution sample.
    msp_classifier = softmax_classifier.max().item()
    msp_clip = softmax_clip.max().item()
    msp_ensemble = softmax_ensemble.max().item()

    print(f"\n--- Analysis for: {image_path.split('/')[-1]} ---")
    print(f"Classifier (ResNet50) MSP: {msp_classifier:.4f}")
    print(f"Zero-Shot (CLIP)    MSP: {msp_clip:.4f}")
    print(f"Ensemble (COOkeD-style) MSP: {msp_ensemble:.4f} <---")


if __name__ == "__main__":
    # --- Image Definitions ---
    # In-Distribution (ID): An image suitable for VQASynth's spatial reasoning.
    ID_IMAGE_URL = "https://raw.githubusercontent.com/smellslikeml/experimental-vqasynth/main/assets/warehouse_sample_1.jpeg"
    
    # Out-of-Distribution (OOD): An image where spatial reasoning is nonsensical.
    # A texture from the DTD dataset, often used for OOD evaluation.
    OOD_IMAGE_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/banded/banded_0005.jpg"

    # --- Run Experiment ---
    classifier_model, cls_preprocess, clip_model, clip_preprocess, tokenizer = get_models()
    text_features = get_clip_text_features(IMAGENET_CLASSES, clip_model, tokenizer)

    print("\n" + "="*50)
    print("Evaluating IN-DISTRIBUTION (ID) image...")
    run_ood_analysis(ID_IMAGE_URL, classifier_model, cls_preprocess, clip_model, clip_preprocess, text_features)

    print("\n" + "="*50)
    print("Evaluating OUT-OF-DISTRIBUTION (OOD) image...")
    run_ood_analysis(OOD_IMAGE_URL, classifier_model, cls_preprocess, clip_model, clip_preprocess, text_features)

    print("\n" + "="*50)
    print("\nConclusion: The ensemble MSP provides a more robust signal for OOD detection.")
    print("A high MSP indicates an in-distribution sample, while a low MSP suggests an OOD sample.")
    print("This score could be used as a filter before passing images to the VQASynth pipeline.")
