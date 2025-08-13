import argparse
import numpy as np
from PIL import Image
from scipy.fft import dctn
import os


def preprocess_image(image_path, size=(96, 96)):
    """Load, convert to grayscale, and resize the image."""
    with Image.open(image_path) as img:
        # Convert to grayscale 'L' mode
        img_gray = img.convert("L")
        # Resize
        img_resized = img_gray.resize(size, Image.Resampling.LANCZOS)
        # Convert to numpy array and normalize to [0, 1]
        return np.array(img_resized) / 255.0


def extract_dct_features(image_array, k, p):
    """
    Compress an image array using 2D-DCT, truncation, and sparsification.
    Inspired by the SCOPE-for-Atari project.
    """
    # 1. Apply 2-D Discrete Cosine Transform
    dct_matrix = dctn(image_array, type=2, norm="ortho")

    # 2. Keep only the top-left KxK block of low-frequency coefficients
    truncated_dct = dct_matrix[:k, :k]

    # 3. Sparsify by zeroing the lowest p-th percentile of absolute values
    if p > 0 and p < 100:
        # Get the threshold value at the p-th percentile of the absolute coefficients
        threshold = np.percentile(np.abs(truncated_dct), p)
        # Zero out coefficients with absolute values below the threshold
        truncated_dct[np.abs(truncated_dct) < threshold] = 0.0

    # 4. Flatten the KxK matrix into a feature vector
    return truncated_dct.flatten()


def main():
    parser = argparse.ArgumentParser(
        description="Extract compact DCT features from an image, inspired by SCOPE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-path", required=True, help="Path to the input image file."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to save the output .npy feature vector.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Size of the KxK block of DCT coefficients to keep.",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=75,
        help="Percentile of smallest (by magnitude) coefficients to zero out for sparsification.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=96,
        help="Standard size to which images are resized before processing.",
    )

    args = parser.parse_args()

    print(f"Processing image: {args.image_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Preprocess the image
    image_array = preprocess_image(args.image_path, size=(args.img_size, args.img_size))

    # Extract DCT features
    features = extract_dct_features(image_array, args.k, args.p)

    # Save features to a .npy file
    np.save(args.output_path, features)
    print(f"Saved {features.shape} DCT feature vector to {args.output_path}")


if __name__ == "__main__":
    main()
