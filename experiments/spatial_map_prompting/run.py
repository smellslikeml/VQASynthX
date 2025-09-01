import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json


def generate_spatial_map(object_data):
    """
    Generates a 2D scatter plot from object location data.
    Inspired by the programmatic chart generation in the SOURCE repo.

    Args:
        object_data (pd.DataFrame): DataFrame with 'label', 'x', and 'y' columns.

    Returns:
        str: Base64 encoded string of the PNG map image.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(object_data["x"], object_data["y"], s=200)

    # Add labels to each point
    for i, row in object_data.iterrows():
        ax.text(row["x"] + 0.1, row["y"] + 0.1, row["label"], fontsize=12)

    ax.set_title("2D Object Location Map")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.grid(True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal", adjustable="box")

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Encode in base64
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return image_base64


def create_multimodal_prompt(question, original_image_b64, map_image_b64):
    """
    Constructs the final JSON prompt for the VLM.
    """
    prompt = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are provided with two images. The first is a photo of a scene. The second is a 2D map showing the relative (x, y) positions of key objects in the scene. Use both images to answer the following question.\n\nQuestion: {question}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{original_image_b64}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{map_image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    return json.dumps(prompt, indent=2)


if __name__ == "__main__":
    # 1. Sample data, representing output from a prior VQASynth stage.
    # This would typically come from a file or another process.
    data = {
        "label": ["Red Forklift", "Cardboard Boxes", "Man in Hat"],
        "x": [2, 7, 6],
        "y": [3, 4, 8],
    }
    object_locations_df = pd.DataFrame(data)

    # 2. A placeholder for the original scene image.
    # In a real run, this would be the actual image from the dataset.
    placeholder_original_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    # 3. Generate the 2D spatial map visualization.
    map_b64 = generate_spatial_map(object_locations_df)

    # 4. Define a sample VQA question.
    vqa_question = (
        "Does the red forklift appear on the left side of the cardboard boxes?"
    )

    # 5. Create the final multimodal prompt.
    final_prompt_json = create_multimodal_prompt(
        vqa_question, placeholder_original_image_b64, map_b64
    )

    # 6. Print the result. This would be saved or passed to the next stage.
    print(final_prompt_json)
