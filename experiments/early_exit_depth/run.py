import torch
import time
import requests
from PIL import Image
import numpy as np
import os

# --- Configuration ---
# URL of a sample image from the VQASynth README
IMAGE_URL = "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
IMAGE_PATH = "warehouse_sample.jpeg"
OUTPUT_DIR = "output"

# Heuristic threshold for uncertainty. If the standard deviation of the fast depth map
# is above this value, we use the more accurate model. This is a simple proxy for scene complexity.
UNCERTAINTY_THRESHOLD = 0.2

# --- Model Loading ---

def get_device():
    """Gets the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_depth_models(device):
    """
    Loads a fast, lightweight depth model and a more accurate, heavier model.
    This simulates the early-exit vs. full-path execution from GFNet-Dynn.
    """
    print("Loading depth estimation models...")
    # Fast Model (e.g., MiDaS small) - The "early exit"
    fast_model_type = "MiDaS_small"
    fast_model = torch.hub.load("intel-isl/MiDaS", fast_model_type, trust_repo=True)
    fast_model.to(device)
    fast_model.eval()

    # Accurate Model (e.g., MiDaS DPT-Large) - The "full network"
    accurate_model_type = "DPT_Large"
    accurate_model = torch.hub.load("intel-isl/MiDaS", accurate_model_type, trust_repo=True)
    accurate_model.to(device)
    accurate_model.eval()

    # Load MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    fast_transform = midas_transforms.small_transform
    accurate_transform = midas_transforms.dpt_transform

    print("Models loaded successfully.")
    return (fast_model, fast_transform), (accurate_model, accurate_transform)

# --- Core Logic ---

def estimate_depth(model, transform, image, device):
    """Runs a single depth estimation model."""
    input_batch = transform(image).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

def check_uncertainty(depth_map):
    """
    A simple heuristic to check if the depth prediction is uncertain.
    Inspired by the confidence check at each exit in an early-exit network.
    """
    # Normalize depth map to [0, 1] for consistent std dev calculation
    normalized_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    std_dev = np.std(normalized_map)
    print(f"Prediction uncertainty (std dev): {std_dev:.4f}")
    return std_dev > UNCERTAINTY_THRESHOLD

def save_depth_map(depth_map, path):
    """Saves the depth map as a grayscale image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Normalize and convert to an image
    output_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    output_image = Image.fromarray((output_normalized * 255).astype(np.uint8))
    output_image.save(path)
    print(f"Saved depth map to {path}")

# --- Main Experiment ---

def main():
    """
    Main function to run the early-exit depth estimation experiment.
    This experiment tests the core idea from GFNet-Dynn: use a fast path for
    easy inputs and a full, slower path for more complex ones.
    """
    print("--- Starting Early-Exit Depth Estimation Experiment ---")
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    (fast_model, fast_transform), (accurate_model, accurate_transform) = load_depth_models(device)

    # Download sample image if it doesn't exist
    if not os.path.exists(IMAGE_PATH):
        print(f"Downloading sample image from {IMAGE_URL}...")
        response = requests.get(IMAGE_URL)
        response.raise_for_status()
        with open(IMAGE_PATH, 'wb') as f:
            f.write(response.content)

    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    # --- Early-Exit Cascade ---
    
    # 1. Run the "early exit" (fast model)
    print("\nStep 1: Running fast model (early exit)...")
    start_time_fast = time.time()
    fast_depth_map = estimate_depth(fast_model, fast_transform, img, device)
    end_time_fast = time.time()
    print(f"Fast model inference time: {end_time_fast - start_time_fast:.4f} seconds.")
    save_depth_map(fast_depth_map, os.path.join(OUTPUT_DIR, "depth_fast.png"))

    # 2. Check confidence and decide whether to proceed
    is_uncertain = check_uncertainty(fast_depth_map)

    # 3. Conditionally run the "full path" (accurate model)
    if is_uncertain:
        print(f"\nResult is uncertain (threshold: {UNCERTAINTY_THRESHOLD}). Proceeding to accurate model.")
        print("Step 2: Running accurate model (full path)...")
        start_time_accurate = time.time()
        final_depth_map = estimate_depth(accurate_model, accurate_transform, img, device)
        end_time_accurate = time.time()
        print(f"Accurate model inference time: {end_time_accurate - start_time_accurate:.4f} seconds.")
        total_time = (end_time_fast - start_time_fast) + (end_time_accurate - start_time_accurate)
        print(f"Total cascaded inference time: {total_time:.4f} seconds.")
        save_depth_map(final_depth_map, os.path.join(OUTPUT_DIR, "depth_final_accurate.png"))
    else:
        print(f"\nResult is confident. Using fast model's output.")
        final_depth_map = fast_depth_map
        total_time = end_time_fast - start_time_fast
        print(f"Total inference time: {total_time:.4f} seconds.")
        save_depth_map(final_depth_map, os.path.join(OUTPUT_DIR, "depth_final_fast.png"))
    
    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    main()
