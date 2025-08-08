import cv2
import numpy as np
import os

def polar_transform(image):
    """
    Applies a polar transformation to an image.
    Inspired by cartesian_to_polar_paper_v06_shannon.py from C2G-KD.
    The output's y-axis corresponds to radius, and x-axis corresponds to angle.
    """
    # The source script uses cv2.warpPolar, which is efficient.
    # We adapt it for general-purpose image crops.
    if len(image.shape) > 2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    height, width = image_gray.shape
    center = (width / 2, height / 2)
    
    # Calculate radius to map to the output image width
    max_radius = np.sqrt((height/2)**2 + (width/2)**2)

    # The size of the polar image
    polar_size = (width, height) 

    polar_image = cv2.warpPolar(
        image_gray,
        polar_size,
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR
    )
    
    return polar_image.astype(np.float32) / 255.0


def main():
    """
    Main function to run the experiment.
    Loads an image, crops an object, applies polar transform, and saves results.
    """
    # This script should be run from the root of the experimental-vqasynth repo.
    # Create an experiment directory for outputs.
    output_dir = "experiments/polar-transform"
    os.makedirs(output_dir, exist_ok=True)

    image_path = "assets/warehouse_sample_1.jpeg"
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found at '{image_path}'.")
        print("Creating a dummy 1200x800 black image with a red rectangle for demonstration.")
        sample_image = np.zeros((800, 1200, 3), dtype=np.uint8)
        # Draw a red rectangle to simulate the forklift
        cv2.rectangle(sample_image, (100, 400), (500, 700), (0, 0, 255), -1) 
    else:
        sample_image = cv2.imread(image_path)

    # Approximate bounding box for the red forklift in warehouse_sample_1.jpeg
    # format: (x_min, y_min, x_max, y_max)
    bbox = (100, 400, 500, 700) 

    # Crop the object
    x1, y1, x2, y2 = bbox
    object_crop = sample_image[y1:y2, x1:x2]
    
    crop_path = os.path.join(output_dir, "output_object_crop.png")
    cv2.imwrite(crop_path, object_crop)
    print(f"Saved object crop to '{crop_path}'")

    # Apply the polar transform from the source repository's concept
    polar_representation = polar_transform(object_crop)
    
    polar_path = os.path.join(output_dir, "output_object_polar.png")
    cv2.imwrite(polar_path, (polar_representation * 255).astype(np.uint8))
    print(f"Saved polar representation to '{polar_path}'")
    print("\nExperiment complete. Check the 'experiments/polar-transform' directory for outputs.")


if __name__ == "__main__":
    main()
