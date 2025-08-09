import torch
import torch.nn as nn
import timm
from PIL import Image
import requests
import torchvision.transforms as T
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Model Definition ---
class LightweightDepthEstimator(nn.Module):
    """
    A lightweight depth estimation model using a MobileViT backbone.
    This architecture is inspired by the findings in lcnn-opt, prioritizing
    efficient, real-time-capable models. It combines a pre-trained image
    classification backbone with a simple upsampling head for depth prediction.
    """
    def __init__(self, backbone_name='mobilevit_s', pretrained=True):
        super().__init__()
        # Load the pre-trained backbone using timm, a library acknowledged in the source repo
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        
        # Get the output channel dimensions from the backbone
        feature_info = self.backbone.feature_info.channels()
        
        # A simple upsampling head to predict the depth map
        # This is a basic example; a more complex decoder (e.g., UNet-style) could be used
        self.head = nn.Sequential(
            nn.Conv2d(feature_info[-1], feature_info[-1] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feature_info[-1] // 2, feature_info[-1] // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feature_info[-1] // 4, 1, kernel_size=1) # Output one channel for depth
        )

    def forward(self, x):
        # The VQASynth pipeline can use various image sizes, so we make the model flexible
        input_h, input_w = x.shape[-2:]
        features = self.backbone(x)
        # We take the highest-level feature map for this simple decoder
        depth_map = self.head(features[-1])
        # Upsample to original image size for comparison
        depth_map = nn.functional.interpolate(
            depth_map, size=(input_h, input_w), mode='bilinear', align_corners=False
        )
        return depth_map

def run_experiment():
    """
    Main function to run the depth estimation experiment.
    """
    print("--- Starting Lightweight Depth Estimation Experiment ---")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print("Loading MobileViT-S based depth model...")
    # The source repo paper shows MobileViT v2 (S) achieving 89.45% Top-1 accuracy.
    model = LightweightDepthEstimator(backbone_name='mobilevit_s').to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Prepare Input Data ---
    # Use a sample image from the VQASynth README
    img_url = "https://github.com/smellslikeml/experimental-vqasynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
    try:
        image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        print(f"Loaded sample image from {img_url}")
    except Exception as e:
        print(f"Failed to load image, creating a dummy tensor. Error: {e}")
        image = Image.new('RGB', (512, 512), color = 'red')

    # Pre-processing transforms similar to standard vision models
    transform = T.Compose([
        T.Resize((384, 384)), # MobileViT-S works well with this size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    # --- 3. Run Inference and Benchmark ---
    print("Benchmarking inference speed...")
    warmup_runs = 5
    benchmark_runs = 20
    timings = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(input_tensor)
        if str(device) == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        for i in range(benchmark_runs):
            start_time = time.time()
            output = model(input_tensor)
            if str(device) == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)

    avg_time = np.mean(timings)
    fps = 1 / avg_time
    print("\n--- Benchmark Results ---")
    print(f"Average Inference Time: {avg_time * 1000:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")

    # --- 4. Visualize Output ---
    print("Visualizing depth map...")
    output_depth = output.squeeze().cpu().numpy()
    
    # Normalize for visualization
    output_depth = (output_depth - output_depth.min()) / (output_depth.max() - output_depth.min())

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image.resize((384, 384)))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_depth, cmap='inferno')
    plt.title("Predicted Depth Map (MobileViT-S Backbone)")
    plt.axis('off')
    
    output_path = "depth_estimation_result.png"
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
