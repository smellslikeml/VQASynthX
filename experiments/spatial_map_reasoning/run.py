import os
import base64
import io
from openai import OpenAI
import matplotlib.pyplot as plt
import requests

# --- Configuration ---
# This script requires the OpenAI Python library and an API key.
# pip install openai matplotlib requests
# export OPENAI_API_KEY='your-api-key'

MODEL_NAME = "gpt-4o"
API_KEY = os.environ.get("OPENAI_API_KEY")

# --- Sample Data (Inspired by VQASynth) ---
# This represents a simplified version of data that could be extracted
# by the VQASynth pipeline.
IMAGE_URL = "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
QUESTION = "Does the red forklift in the warehouse appear on the left side of the brown cardboard boxes?"

# Simplified 2D coordinates of objects in a top-down view.
# In a real scenario, these would be derived from the VQASynth pipeline.
OBJECT_COORDINATES = {
    "red forklift": (2, 5),
    "brown boxes": (6, 5),
    "blue container": (9, 2),
}

# --- Core Logic ---

def image_to_base64(image_source):
    """Converts an image from a URL or local path to a base64 string."""
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source)
        response.raise_for_status()
        image_bytes = response.content
    else:
        with open(image_source, "rb") as image_file:
            image_bytes = image_file.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def generate_2d_map(objects, output_path="map.png"):
    """
    Generates a simple 2D scatter plot representing a top-down map of objects.
    This is inspired by the visualizations used in the lvlm-vis-data-understanding study.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot objects
    for name, (x, y) in objects.items():
        ax.scatter(x, y, s=200, label=name, alpha=0.8)
        ax.text(x, y + 0.5, name, ha='center', va='bottom', fontsize=9)

    # Style the plot
    ax.set_title("Top-Down Object Map")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # Save to a bytes buffer and then to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    with open(output_path, 'wb') as f:
        f.write(buf.read())
        
    print(f"Generated map at {output_path}")
    return output_path

def run_vqa_query(client, question, image_url, map_path=None):
    """
    Runs a query against a Vision Language Model with or without a supplementary map.
    """
    print("\n" + "="*50)
    if map_path:
        print(f"Running query WITH 2D map: {map_path}")
    else:
        print("Running query with image ONLY")
    print("="*50)

    base64_image = image_to_base64(image_url)
    
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
    ]
    
    if map_path:
        base64_map = image_to_base64(map_path)
        content.append({
            "type": "text",
            "text": "Here is a simplified 2D top-down map of the key objects in the scene:"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_map}"}
        })
        
    content.append({
        "type": "text",
        "text": f"Question: {question}. Please answer the question and explain your reasoning based on the image(s) provided."
    })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=300
        )
        print("VLM Response:")
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    """
    Main function to run the experiment.
    """
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=API_KEY)

    # --- Run Experiment ---

    # 1. Generate the 2D map visualization
    map_file_path = generate_2d_map(OBJECT_COORDINATES)

    # 2. Query the VLM with the image only
    run_vqa_query(client, QUESTION, IMAGE_URL)

    # 3. Query the VLM with both the image and the 2D map
    run_vqa_query(client, QUESTION, IMAGE_URL, map_path=map_file_path)

if __name__ == "__main__":
    main()
