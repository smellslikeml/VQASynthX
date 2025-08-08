import os
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Helper Functions Adapted from SOURCE ---
# Source: C2G-KD, cartesian_to_polar_paper_v06_shannon.py

def polar_transform(image_2d, output_size=(128, 128)):
    """
    Transforms a 2D Cartesian image to a polar representation using OpenCV.
    """
    if image_2d.ndim > 2:
        image_2d = cv2.cvtColor(image_2d, cv2.COLOR_BGR2GRAY)
    
    # Ensure image is in uint8 format for OpenCV
    if image_2d.dtype != np.uint8:
        image_2d_uint8 = (image_2d * 255).astype(np.uint8)
    else:
        image_2d_uint8 = image_2d

    height, width = image_2d.shape[:2]
    center = (width / 2, height / 2)
    max_radius = np.sqrt((height/2)**2 + (width/2)**2)

    polar_image = cv2.warpPolar(
        image_2d_uint8,
        output_size,
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR
    )
    return polar_image.astype(np.float32) / 255.0

def polar_to_cartesian(polar_img, cartesian_size=(128, 128)):
    """
    Transforms a polar image back to a 2D Cartesian representation.
    """
    if polar_img.dtype != np.uint8:
        polar_img_uint8 = np.clip(polar_img * 255, 0, 255).astype(np.uint8)
    else:
        polar_img_uint8 = polar_img
        
    height, width = cartesian_size
    center = (width / 2, height / 2)
    max_radius = np.sqrt((height/2)**2 + (width/2)**2)

    cartesian_img = cv2.warpPolar(
        polar_img_uint8,
        cartesian_size,
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return cartesian_img.astype(np.float32) / 255.0

# --- Main Experiment ---

def run_experiment():
    """
    Runs the polar-PCA reconstruction experiment.
    """
    # Configuration
    INPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'warehouse_sample_1.jpeg')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Coordinates to crop the red forklift (approximate)
    CROP_BOX = (50, 450, 450, 750) # (left, top, right, bottom)
    PATCH_SIZE = (128, 128)
    N_COMPONENTS = 32

    # 1. Load and prepare image patch
    print(f"Loading image from {INPUT_IMAGE_PATH}")
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERROR: Input image not found at {INPUT_IMAGE_PATH}")
        print("Please ensure you are running this from the root of the 'experimental-vqasynth' repository.")
        return

    image = Image.open(INPUT_IMAGE_PATH).convert('L') # Grayscale
    patch = image.crop(CROP_BOX)
    patch = patch.resize(PATCH_SIZE, Image.Resampling.LANCZOS)
    patch_np = np.array(patch).astype(np.float32) / 255.0

    # 2. Apply Polar Transform
    print("Applying polar transform...")
    polar_patch = polar_transform(patch_np, output_size=PATCH_SIZE)

    # 3. Apply PCA for dimensionality reduction and reconstruction
    # We fit and transform the polar image itself to test the reconstruction.
    print(f"Applying PCA with {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS)
    transformed_polar = pca.fit_transform(polar_patch)
    reconstructed_polar = pca.inverse_transform(transformed_polar)

    # 4. Convert reconstructed polar image back to Cartesian
    print("Converting back to Cartesian coordinates...")
    reconstructed_cartesian = polar_to_cartesian(reconstructed_polar, cartesian_size=PATCH_SIZE)

    # 5. Visualize and save results
    print("Generating and saving comparison plot...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(patch_np, cmap='gray')
    axes[0].set_title('Original Cropped Object')
    axes[0].axis('off')

    axes[1].imshow(polar_patch, cmap='gray', aspect='auto')
    axes[1].set_title('Polar Representation')
    axes[1].axis('off')

    axes[2].imshow(reconstructed_polar, cmap='gray', aspect='auto')
    axes[2].set_title(f'Reconstructed Polar (PCA n={N_COMPONENTS})')
    axes[2].axis('off')

    axes[3].imshow(reconstructed_cartesian, cmap='gray')
    axes[3].set_title('Reconstructed Cartesian')
    axes[3].axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, 'reconstruction_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    run_experiment()
