import argparse
import pathlib
import cv2
import numpy as np
from PIL import Image


def apply_polar_transform(
    image: np.ndarray, output_size: tuple = (256, 256)
) -> np.ndarray:
    """
    Applies a polar coordinate transformation to an image.
    Inspired by the polar transform used in the C2G-KD repository for feature analysis.

    Args:
        image (np.ndarray): The input image as a NumPy array (H, W, C).
        output_size (tuple): The desired (width, height) of the polar image.

    Returns:
        np.ndarray: The polar-transformed image.
    """
    # Convert to grayscale if it has color channels
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure image is in uint8 format for OpenCV
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Max radius is half the smaller dimension to stay within the image
    max_radius = min(width, height) / 2

    # Use cv2.warpPolar for a standard and efficient implementation
    # Flags: WARP_POLAR_LINEAR for linear interpolation
    polar_image = cv2.warpPolar(
        image,
        dsize=output_size,
        center=center,
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LINEAR,
    )

    return polar_image


def main():
    """
    Main function to parse arguments and run the polar transform process.
    """
    parser = argparse.ArgumentParser(
        description="Apply a polar coordinate transformation to an image. This stage is an "
        "experiment inspired by C2G-KD to explore alternative spatial feature "
        "representations for VQA synthesis."
    )
    parser.add_argument(
        "--input_image",
        type=pathlib.Path,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Directory to save the transformed image.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=256,
        help="The size (width and height) of the output polar image.",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    try:
        # Handle potential RGBA images from assets
        img_pil = Image.open(args.input_image).convert("RGB")
        image = np.array(img_pil)
    except Exception as e:
        print(f"Error loading image {args.input_image}: {e}")
        return

    # Apply transform
    polar_transformed_image = apply_polar_transform(
        image, output_size=(args.output_size, args.output_size)
    )

    # Save result
    output_filename = f"{args.input_image.stem}_polar.png"
    output_path = args.output_dir / output_filename

    try:
        Image.fromarray(polar_transformed_image).save(output_path)
        print(f"Successfully saved polar transformed image to {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")


if __name__ == "__main__":
    main()
