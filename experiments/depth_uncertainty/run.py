import torch
import numpy as np
import requests
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor
import matplotlib.pyplot as plt
import os

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def run_mc_dropout_depth_estimation(image_url, n_samples=15, output_dir="output/depth_uncertainty"):
    """
    Performs depth estimation using Monte Carlo Dropout to quantify uncertainty.

    Args:
        image_url (str): URL of the input image.
        n_samples (int): Number of forward passes for MC Dropout.
        output_dir (str): Directory to save the output plots.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image from URL
    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Load a pretrained depth estimation model and processor
    model_name = "Intel/dpt-hybrid-midas"
    processor = DPTImageProcessor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name)
    
    # Enable dropout layers for uncertainty estimation. 
    # The DPT model has dropout layers in its ViT backbone.
    enable_dropout(model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # Store predictions from each forward pass
    predictions = []

    print(f"Running {n_samples} forward passes for MC Dropout...")
    with torch.no_grad():
        for i in range(n_samples):
            print(f"  - Pass {i+1}/{n_samples}")
            # Prepare image for the model
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Predict depth
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            predictions.append(prediction.cpu())

    # Stack predictions and calculate mean and variance
    predictions_tensor = torch.stack(predictions)
    mean_depth = predictions_tensor.mean(axis=0)
    variance_depth = predictions_tensor.var(axis=0)

    print("Generating and saving plots...")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # Mean Depth
    im1 = axes[1].imshow(mean_depth.numpy(), cmap="plasma")
    axes[1].set_title("Mean Depth Prediction")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.05, pad=0.1)

    # Uncertainty (Variance)
    im2 = axes[2].imshow(variance_depth.numpy(), cmap="magma")
    axes[2].set_title("Uncertainty (Variance)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], orientation='horizontal', fraction=0.05, pad=0.1)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "depth_uncertainty_analysis.png")
    plt.savefig(output_path)
    print(f"Saved analysis to {output_path}")

if __name__ == "__main__":
    # Using one of the sample images from the VQASynth README
    sample_image_url = "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
    run_mc_dropout_depth_estimation(sample_image_url)
