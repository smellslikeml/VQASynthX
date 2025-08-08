import base64
import io
import json
import re
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "remyxai/SpaceThinker-Qwen2.5VL-3B"
DATASET_ID = "remyxai/SpaceOm"
NUM_SAMPLES = 20 # Keep it small for a minimal test

def generate_2d_map(objects):
    """
    Generates a top-down 2D scatter plot of object locations.
    Adapted from the lvlm-vis-data-understanding repo.
    Assumes VQASynth provides 3D coordinates where x and z form the top-down plane.
    """
    if not objects:
        return None

    # VQASynth uses a different coordinate system, we assume 'x' and 'z' are top-down.
    try:
        x_coords = [obj['center_3d'][0] for obj in objects]
        y_coords = [obj['center_3d'][2] for obj in objects]
        labels = [f"Obj {i+1}" for i in range(len(objects))]
    except (KeyError, IndexError):
        # Return None if the data format is unexpected
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x_coords, y_coords, zorder=2)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,5), ha='center')
    
    ax.set_title("Top-Down Object Map")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate (Depth)")
    ax.grid(True, zorder=1)
    ax.set_aspect('equal', adjustable='box')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img_bytes = buf.getvalue()
    return Image.open(io.BytesIO(img_bytes))

def parse_distance(text):
    """
    Parses the first numeric value from the model's output string.
    """
    match = re.search(r'(\d+\.?\d*)', text)
    return float(match.group(1)) if match else None

def main():
    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Load dataset
    dataset = load_dataset(DATASET_ID, split="train")
    
    results = []
    
    # Select a subset of data that contains ground truth distances
    eval_samples = dataset.filter(lambda x: x.get('distance_meters') is not None).select(range(NUM_SAMPLES))

    for sample in tqdm(eval_samples, desc="Evaluating samples"):
        try:
            image_url = sample['image_url']
            # Assuming the question is the first user turn in conversations
            question = next(turn['value'] for turn in sample['conversations'] if turn['from'] == 'human')
            ground_truth_dist = float(sample['distance_meters'])
            # scene_graph is often stored as a JSON string
            scene_objects = json.loads(sample['scene_graph'])['objects']
        except (KeyError, StopIteration, json.JSONDecodeError):
            continue # Skip samples with unexpected format

        # --- Baseline Case (Image Only) ---
        prompt_baseline = f"USER: <image>\n{question}\nASSISTANT:"
        inputs_baseline = processor(text=prompt_baseline, images=[image_url], return_tensors="pt").to(model.device, torch.float16)
        output_baseline = model.generate(**inputs_baseline, max_new_tokens=128)
        response_baseline_text = processor.decode(output_baseline[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
        dist_baseline = parse_distance(response_baseline_text)

        # --- Experiment Case (Image + 2D Map) ---
        map_image = generate_2d_map(scene_objects)
        if map_image is None:
            dist_experiment = None
            response_experiment_text = "Error: Could not generate map."
        else:
            prompt_experiment = f"USER: <image>\nHere is a top-down 2D map of the detected objects for context: <image>\nBased on the original photo and the map, answer the following question: {question}\nASSISTANT:"
            inputs_experiment = processor(text=prompt_experiment, images=[image_url, map_image], return_tensors="pt").to(model.device, torch.float16)
            output_experiment = model.generate(**inputs_experiment, max_new_tokens=128)
            response_experiment_text = processor.decode(output_experiment[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
            dist_experiment = parse_distance(response_experiment_text)

        results.append({
            "question": question,
            "ground_truth": ground_truth_dist,
            "baseline_pred": dist_baseline,
            "experiment_pred": dist_experiment,
        })
        
    # --- Calculate and Print Metrics ---
    df = pd.DataFrame(results)
    df.dropna(inplace=True)

    if not df.empty:
        mae_baseline = (df['baseline_pred'] - df['ground_truth']).abs().mean()
        mae_experiment = (df['experiment_pred'] - df['ground_truth']).abs().mean()

        print("\n--- Evaluation Complete ---")
        print(f"Processed {len(df)} valid samples.")
        print(f"MAE (Baseline - Image Only): {mae_baseline:.4f}")
        print(f"MAE (Experiment - Image + 2D Map): {mae_experiment:.4f}")
        
        improvement = mae_baseline - mae_experiment
        improvement_percent = (improvement / mae_baseline) * 100 if mae_baseline > 0 else 0
        
        print(f"\nImprovement (lower MAE is better): {improvement:.4f} ({improvement_percent:.2f}%)")
        print("\nSample Results:")
        print(df.head().to_string())
    else:
        print("\n--- Evaluation Failed: No valid samples could be processed ---")

if __name__ == "__main__":
    main()
