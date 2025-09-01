import os
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
OUTPUT_DIR = "output/geospatial_bootstrapping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Stage 1: Load Noisy Geospatial Data ---
# In a real implementation, this would load a large CSV/GeoJSON file,
# similar to RampNet loading data for NYC, Portland, etc.
# For this demo, we'll use a small, hardcoded sample.
NOISY_GEOSPATIAL_DATA = [
    {
        "id": "store_001",
        "name": "Generic Coffee Shop",
        "lat": 47.6062,
        "lon": -122.3321,
    },
    {"id": "store_002", "name": "Local Bookstore", "lat": 47.6065, "lon": -122.3325},
]


def log_step(message):
    """Helper function for logging pipeline steps."""
    print(f"[PIPELINE] {message}")


def fetch_street_level_image(lat, lon, item_id):
    """
    Placeholder for fetching a street-level image (e.g., from Google Street View API).

    RampNet uses Google Street View. For this demo, we create a dummy image.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        item_id (str): Unique identifier for the data point.

    Returns:
        PIL.Image: The fetched (or created) image.
    """
    log_step(f"Fetching image for {item_id} at ({lat}, {lon})...")
    # Create a blank image as a placeholder.
    img = Image.new("RGB", (1024, 512), color="gray")
    d = ImageDraw.Draw(img)
    try:
        # Use a common font if available, otherwise default.
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    d.text(
        (10, 10),
        f"Simulated Street View for {item_id}",
        fill=(255, 255, 255),
        font=font,
    )
    return img


def project_geo_to_image_plane(image, lat, lon):
    """
    Placeholder for projecting a geo-coordinate onto the 2D image.

    This is a complex step in reality, involving camera intrinsics and panorama geometry.
    RampNet's pipeline performs this to get an initial, noisy pixel coordinate.
    Here, we'll just return a fixed, noisy bounding box for demonstration.

    Args:
        image (PIL.Image): The image to project onto.
        lat (float): Latitude.
        lon (float): Longitude.

    Returns:
        tuple: A noisy bounding box (x1, y1, x2, y2).
    """
    log_step("Projecting geo-coordinate to initial noisy bounding box...")
    # Simulate a noisy projection by placing a box somewhere on the image.
    w, h = image.size
    x1, y1 = w * 0.4, h * 0.4
    x2, y2 = w * 0.6, h * 0.7
    return (x1, y1, x2, y2)


def refine_location_with_model(image, noisy_bbox):
    """
    Placeholder for a refinement model.

    RampNet trains a "crop-level model" on a small, manually labeled dataset.
    This model takes a crop of the image around the noisy projection and
    outputs a more precise location, or confirms the object isn't present.

    Args:
        image (PIL.Image): The full image.
        noisy_bbox (tuple): The initial noisy bounding box.

    Returns:
        tuple or None: A refined bounding box (x1, y1, x2, y2) or None if not found.
    """
    log_step("Refining location with a simulated crop-level model...")
    # Simulate the model slightly adjusting the box and confirming its presence.
    x1, y1, x2, y2 = noisy_bbox
    refined_bbox = (x1 + 15, y1 + 10, x2 + 5, y2 + 20)  # Simulate a small shift

    # The model could also return None if the object is not detected in the crop.
    # We'll always return a successful detection for this demo.
    return refined_bbox


def save_labeled_data(image, final_bbox, item_id):
    """
    Saves the image with the final bounding box drawn, and the label data.

    This represents the final output of the data synthesis pipeline: an image
    and its corresponding high-quality label, ready for training a detector.

    Args:
        image (PIL.Image): The original image.
        final_bbox (tuple): The refined bounding box.
        item_id (str): The unique ID for the data point.
    """
    log_step(f"Saving final labeled data for {item_id}...")

    # Save label file (e.g., in a simple text format)
    label_path = os.path.join(OUTPUT_DIR, f"{item_id}.txt")
    with open(label_path, "w") as f:
        # Simple format: class_id, x1, y1, x2, y2
        class_id = 0  # Assuming a single class for this object type
        f.write(f"{class_id} {' '.join(map(str, [int(c) for c in final_bbox]))}\n")

    # Save visualization image
    draw = ImageDraw.Draw(image)
    draw.rectangle(final_bbox, outline="green", width=3)
    image_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")
    image.save(image_path)
    log_step(f"Saved label to {label_path} and image to {image_path}")


def main():
    """
    Main function to run the RampNet-inspired data synthesis pipeline.
    """
    log_step("Starting geospatial data bootstrapping pipeline...")

    for item in NOISY_GEOSPATIAL_DATA:
        print("\n" + "=" * 40)
        log_step(f"Processing item: {item['id']} ({item['name']})")

        # 1. Fetch street-level image for the geo-coordinate.
        image = fetch_street_level_image(item["lat"], item["lon"], item["id"])

        # 2. Project geo-coordinate to get a noisy initial label.
        noisy_bbox = project_geo_to_image_plane(image, item["lat"], item["lon"])

        # 3. Use a refinement model to get a high-quality label.
        final_bbox = refine_location_with_model(image, noisy_bbox)

        if final_bbox:
            # 4. Save the image and its new label to create the training dataset.
            save_labeled_data(image.copy(), final_bbox, item["id"])
        else:
            log_step(
                f"Object not found for item {item['id']} after refinement. Skipping."
            )

    print("\n" + "=" * 40)
    log_step("Pipeline finished.")
    log_step(f"Generated data can be found in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
