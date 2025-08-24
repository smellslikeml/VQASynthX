import os
import argparse
import numpy as np
import cv2
import pandas as pd
from PIL import Image


def refine_mask_with_bbox(mask_path, bbox):
    """
    Refines a segmentation mask by keeping only the parts within a bounding box.

    Args:
        mask_path (str): Path to the noisy pseudo-label mask file.
        bbox (tuple): A tuple of (xmin, ymin, xmax, ymax) for the bounding box.

    Returns:
        numpy.ndarray: The refined mask as a NumPy array.
    """
    try:
        # Load the noisy mask, ensure it's grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask at {mask_path}. Skipping.")
            return None

        # Create a blank mask with the same dimensions
        refined_mask = np.zeros_like(mask)

        # Unpack bounding box
        xmin, ymin, xmax, ymax = bbox

        # Define the region of interest (ROI) from the bounding box
        roi = mask[int(ymin) : int(ymax), int(xmin) : int(xmax)]

        # Copy the ROI from the original mask to the blank mask
        refined_mask[int(ymin) : int(ymax), int(xmin) : int(xmax)] = roi

        return refined_mask

    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return None


def main(args):
    """
    Main function to process a dataset of images and masks with weak labels.
    """
    print("Starting pseudo-label refinement process...")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # Load weak labels
    try:
        weak_labels_df = pd.read_csv(args.weak_labels)
        print(f"Loaded {len(weak_labels_df)} weak labels from {args.weak_labels}")
    except FileNotFoundError:
        print(f"Error: Weak labels file not found at {args.weak_labels}")
        return

    # Process each entry in the weak labels file
    for index, row in weak_labels_df.iterrows():
        image_name = row["image_name"]

        # In this experiment, the mask has the same base name as the image
        mask_filename = os.path.splitext(image_name)[0] + ".png"
        mask_path = os.path.join(args.mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(
                f"Warning: Mask file not found for {image_name} at {mask_path}. Skipping."
            )
            continue

        bbox = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])

        print(f"Processing {image_name} with bbox {bbox}...")

        # Refine the mask
        refined_mask_array = refine_mask_with_bbox(mask_path, bbox)

        if refined_mask_array is not None:
            # Save the refined mask
            output_path = os.path.join(args.output_dir, mask_filename)
            # Use Pillow to save to maintain format consistency if needed
            Image.fromarray(refined_mask_array).save(output_path)
            print(f"Saved refined mask to {output_path}")

    print("Refinement process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Refine noisy pseudo-label segmentation masks using weak labels (bounding boxes).
        This script simulates the human-in-the-loop refinement step from CEDANet,
        applying it to improve segmentation data for VQA synthesis pipelines.
        """
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory containing the noisy pseudo-label masks (e.g., from a base segmentation model).",
    )
    parser.add_argument(
        "--weak_labels",
        type=str,
        required=True,
        help="Path to a CSV file with weak labels. Must contain columns: 'image_name', 'xmin', 'ymin', 'xmax', 'ymax'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the refined masks.",
    )

    # In a real scenario, you might pass image_dir too, but for this minimal example,
    # we only need the masks and their corresponding weak labels.

    args = parser.parse_args()
    main(args)
