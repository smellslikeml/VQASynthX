import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def apply_anxiety_tunnel_vision(
    image: Image.Image, severity: float = 0.8
) -> Image.Image:
    """
    Applies a filter to simulate anxiety-induced tunnel vision.

    This effect is achieved by desaturating and darkening the periphery of the image,
    drawing focus to the center. The severity parameter controls the intensity of the effect.
    This implementation is inspired by the condition described in the Perceptual Reality
    Transformer (https://github.com/linlab/prt) paper.

    Args:
        image (Image.Image): The input PIL image.
        severity (float): The intensity of the effect, from 0.0 (no effect) to 1.0 (max effect).

    Returns:
        Image.Image: The perturbed PIL image.
    """
    if severity == 0:
        return image

    width, height = image.size
    hsv_image = image.convert("HSV")
    h, s, v = hsv_image.split()

    # Create a radial gradient mask
    center_x, center_y = width / 2, height / 2
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Normalize distance to be 0 at center and 1 at the furthest corner
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_dist = dist_from_center / max_dist

    # Invert the gradient: 1 at the center, falling off to 0
    focus_mask = 1 - normalized_dist

    # Apply severity scaling to control the size of the focused area and sharpness of falloff
    focus_mask = np.clip(focus_mask * (2.0 - severity * 1.5), 0, 1)

    # Convert numpy mask to PIL Image for blending
    mask_s = Image.fromarray((focus_mask * 255).astype(np.uint8))
    mask_v = Image.fromarray((focus_mask * 255).astype(np.uint8))

    # Create fully desaturated and darkened layers for blending
    s_desaturated = Image.new("L", (width, height), 0)
    v_darkened = Image.new("L", (width, height), 0)

    # Composite the original saturation/value with the modified versions using the mask
    s_final = Image.composite(s, s_desaturated, mask_s)
    v_final = Image.composite(v, v_darkened, mask_v)

    final_hsv = Image.merge("HSV", (h, s_final, v_final))
    return final_hsv.convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Apply a perceptual anxiety/tunnel vision filter to an image. "
        "Inspired by linlab/prt to test VLM robustness."
    )
    parser.add_argument("input_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "output_path", type=Path, help="Path to save the processed image."
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.8,
        help="Severity of the tunnel vision effect (0.0 to 1.0).",
    )

    args = parser.parse_args()

    if not args.input_path.is_file():
        raise FileNotFoundError(f"Input file not found at {args.input_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading image from: {args.input_path}")
    original_image = Image.open(args.input_path).convert("RGB")

    print(f"Applying anxiety tunnel vision filter with severity {args.severity}...")
    perturbed_image = apply_anxiety_tunnel_vision(original_image, args.severity)

    print(f"Saving perturbed image to: {args.output_path}")
    perturbed_image.save(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
