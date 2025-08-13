import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def create_dummy_data():
    """Creates dummy RGB and depth images for the experiment."""
    # Create a simple scene: a small square closer to the viewer than a large square
    img_size = (256, 256)

    # Dummy RGB image
    rgb_array = np.full((*img_size, 3), 200, dtype=np.uint8)
    rgb_array[50:150, 50:150] = [100, 100, 100]  # Farther, larger square
    rgb_array[100:125, 100:125] = [255, 0, 0]  # Closer, smaller square
    rgb_image = Image.fromarray(rgb_array)
    rgb_image.save("dummy_scene.png")

    # Dummy Depth map (grayscale, darker is closer)
    # Depth values range from 0 (close) to 255 (far)
    depth_array = np.full(img_size, 255, dtype=np.uint8)
    depth_array[50:150, 50:150] = 150  # Farther square
    depth_array[100:125, 100:125] = 50  # Closer square
    depth_image = Image.fromarray(depth_array, "L")
    depth_image.save("dummy_depth.png")

    print("Created dummy_scene.png and dummy_depth.png")
    return "dummy_scene.png", "dummy_depth.png"


def synthesize_attention_from_depth(depth_map_path):
    """
    Synthesizes an attention map from a depth map.
    Assumes closer objects (darker in depth map) get more attention (brighter in attention map).

    Args:
        depth_map_path (str): Path to the depth map image.

    Returns:
        np.ndarray: The synthesized attention map.
    """
    with Image.open(depth_map_path).convert("L") as depth_img:
        depth_array = np.array(depth_img, dtype=np.float32)

    # Invert the depth values so that closer objects (lower values) have higher intensity
    # Adding a small epsilon to avoid division by zero if depth is 0
    attention_map = 1.0 / (depth_array + 1e-6)

    # Normalize to 0-1 range for visualization
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    if max_val > min_val:
        attention_map = (attention_map - min_val) / (max_val - min_val)

    return attention_map


def generate_vqa_pair(attention_map, scene_image_path):
    """Generates a sample VQA pair based on the attention map."""
    # This is a placeholder for a more sophisticated logic.
    # For this example, we'll just generate a generic question.
    question = (
        "In this scene, which area commands the most attention based on proximity?"
    )
    answer = (
        "The small red square in the center, as it is the closest object to the viewer."
    )

    print("\n--- Sample VQA Pair ---")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-----------------------\n")


def visualize_synthesis(rgb_path, depth_path, attention_map):
    """
    Visualizes the original image, depth map, and the synthesized attention map.
    This visualization is inspired by the multi-panel output in SOFA's inference.py.
    """
    rgb_img = np.array(Image.open(rgb_path))
    depth_img = np.array(Image.open(depth_path).convert("L"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Scene")
    axes[0].axis("off")

    axes[1].imshow(depth_img, cmap="gray_r")  # inverted gray for intuitive depth
    axes[1].set_title("Depth Map (Darker=Closer)")
    axes[1].axis("off")

    axes[2].imshow(attention_map, cmap="hot")
    axes[2].set_title("Synthesized Attention Map")
    axes[2].axis("off")

    plt.tight_layout()
    output_path = "attention_synthesis_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("Starting attention map synthesis experiment...")

    # Setup: Create dummy data for a self-contained run
    rgb_path, depth_path = create_dummy_data()

    # Step 1: Synthesize the attention map from the depth map
    # This is analogous to SOFA using procedural data maps.
    print(f"Synthesizing attention map from {depth_path}...")
    attention_map = synthesize_attention_from_depth(depth_path)

    # Step 2: Visualize the inputs and outputs
    # This is inspired by the visualization in SOFA's inference.py
    print("Generating visualization...")
    visualize_synthesis(rgb_path, depth_path, attention_map)

    # Step 3: Generate a sample VQA pair using the new data
    # This connects the synthesized data back to the VQASynth goal.
    print("Generating sample VQA pair...")
    generate_vqa_pair(attention_map, rgb_path)

    # Cleanup
    os.remove(rgb_path)
    os.remove(depth_path)
    print("Experiment complete.")


if __name__ == "__main__":
    main()
