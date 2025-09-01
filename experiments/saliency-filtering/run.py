import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path


def compute_saliency_map(image_path):
    """
    Computes a fine-grained saliency map for an image.
    This serves as a proxy for human attention/gaze data inspired by WebEyeTrack.
    Requires opencv-contrib-python.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Initialize the fine-grained saliency model
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliency_map) = saliency.computeSaliency(img)

    if not success:
        raise RuntimeError("Failed to compute saliency map.")

    # Normalize map to 0-1 range for easier interpretation and scoring
    normalized_map = cv2.normalize(
        saliency_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    return normalized_map


def calculate_bbox_saliency_score(saliency_map, bboxes):
    """
    Calculates the average saliency score for a list of bounding boxes.
    """
    if not bboxes:
        return 0.0

    total_score = 0
    valid_bboxes = 0
    for bbox in bboxes:
        # Assuming bbox is [x_min, y_min, x_max, y_max] in pixel coordinates
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Clamp coordinates to be within map dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(saliency_map.shape[1] - 1, x_max)
        y_max = min(saliency_map.shape[0] - 1, y_max)

        if x_min >= x_max or y_min >= y_max:
            continue  # Skip invalid or zero-area boxes

        roi = saliency_map[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            total_score += np.mean(roi)
            valid_bboxes += 1

    return total_score / valid_bboxes if valid_bboxes > 0 else 0.0


def filter_vqa_by_saliency(vqa_data, image_path, saliency_threshold):
    """
    Filters a list of VQA samples based on the saliency of their mentioned objects.
    """
    saliency_map = compute_saliency_map(image_path)

    filtered_data = []
    for item in vqa_data:
        # The VQASynth pipeline identifies objects; we assume their bboxes are available.
        # This key is hypothetical and would be adapted to the actual VQASynth data schema.
        bboxes = item.get("object_bboxes", [])

        score = calculate_bbox_saliency_score(saliency_map, bboxes)
        item["saliency_score"] = score

        if score >= saliency_threshold:
            filtered_data.append(item)

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter VQA data based on object saliency."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--vqa_json_path",
        type=str,
        required=True,
        help="Path to the VQA JSON data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save filtered data.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Saliency score threshold for filtering.",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading VQA data from {args.vqa_json_path}")
    with open(args.vqa_json_path, "r") as f:
        vqa_samples = json.load(f)

    print(f"Processing image: {args.image_path}")
    print(f"Using saliency threshold: {args.threshold}")

    filtered_samples = filter_vqa_by_saliency(
        vqa_samples, args.image_path, args.threshold
    )

    total_count = len(vqa_samples)
    filtered_count = len(filtered_samples)

    print(f"\nFiltering complete.")
    print(f"Original samples: {total_count}")
    print(
        f"Filtered samples: {filtered_count} ({100 * filtered_count / total_count if total_count > 0 else 0:.2f}%)"
    )

    output_path = os.path.join(
        args.output_dir, f"filtered_{os.path.basename(args.vqa_json_path)}"
    )
    with open(output_path, "w") as f:
        json.dump(filtered_samples, f, indent=2)
    print(f"Filtered data saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    # To test, create a dummy image and a dummy vqa.json:
    # a. Dummy image: a 200x200 black image with a white square at [50,50,100,100]
    # b. Dummy vqa.json:
    #    [
    #      {"id": 1, "question": "q1", "object_bboxes": [[50,50,100,100]]}, // salient
    #      {"id": 2, "question": "q2", "object_bboxes": [[10,10,20,20]]}    // non-salient
    #    ]
    # Command:
    # python run.py --image_path path/to/dummy_image.png --vqa_json_path path/to/vqa.json
    main()
