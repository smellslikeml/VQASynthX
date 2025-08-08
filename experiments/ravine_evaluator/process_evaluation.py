import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO

# A simple regex to find a number (integer or float) possibly followed by units.
# This is a simplified example; a robust solution would handle more unit types and variations.
DISTANCE_REGEX = re.compile(r"(\d+\.?\d*)\s*(?:meters|meter|m|centimeters|cm|feet|ft|inches|in)")

def parse_distance(text):
    """
    Parses a string to find a numerical distance value.
    For this experiment, it simplifies by assuming the first number found is the distance
    and does not perform unit conversion. A full implementation would need to.
    """
    match = DISTANCE_REGEX.search(text.lower())
    if match:
        try:
            # For simplicity, we are not converting units in this minimal script.
            # We are assuming the model is prompted to respond in a consistent unit (e.g., meters).
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

def load_model(model_id):
    """Loads the VLM and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    return model, tokenizer

def run_inference(model, tokenizer, prompt, image_path):
    """Runs inference on a single image and prompt."""
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Process image
    image_tensor = model.vision_tower.image_processor(image, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.bfloat16)
    
    input_ids = model_inputs['input_ids']
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=1024,
            use_cache=True
        )
    
    output_ids = output_ids[:, input_ids.shape[1]:]
    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    
    return response_text


def main(args):
    print(f"Loading model: {args.model_id}")
    model, tokenizer = load_model(args.model_id)

    errors = []
    
    print(f"Loading evaluation data from: {args.eval_file}")
    with open(args.eval_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                image_path = item['image_path']
                prompt = item['prompt']
                ground_truth = float(item['ground_truth_meters'])
                
                print(f"\n--- Processing Sample {i+1} ---")
                print(f"Prompt: {prompt}")

                response_text = run_inference(model, tokenizer, prompt, image_path)
                print(f"Model Response: {response_text}")

                predicted_distance = parse_distance(response_text)
                
                if predicted_distance is not None:
                    error = abs(predicted_distance - ground_truth)
                    errors.append(error)
                    print(f"Ground Truth: {ground_truth:.2f}m | Predicted: {predicted_distance:.2f}m | Absolute Error: {error:.2f}m")
                else:
                    print("Could not parse a valid distance from the response.")

            except Exception as e:
                print(f"Failed to process line {i+1}: {e}")

    if errors:
        mean_absolute_error = sum(errors) / len(errors)
        print("\n--- Evaluation Summary ---")
        print(f"Total samples processed successfully: {len(errors)}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f} meters")
    else:
        print("\n--- Evaluation Summary ---")
        print("No samples were successfully processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantitative evaluation for VQASynth models.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID to evaluate.")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the JSONL file with evaluation data.")
    
    args = parser.parse_args()
    main(args)
