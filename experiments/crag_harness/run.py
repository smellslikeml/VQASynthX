# experiments/crag_harness/run.py

import argparse
import json
from pathlib import Path
import requests
from io import BytesIO

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- Configuration ---
# This configuration defines the model to test and the sample dataset.
# This is inspired by the structured evaluation setup in the Meta CRAG-MM challenge.
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("./eval_data")
IMAGE_URL = "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
IMAGE_PATH = DATA_DIR / "sample_image.jpg"

# Sample evaluation data, mimicking a small slice of a benchmark dataset.
# The structure (image, question, ground_truth) is common in VQA benchmarks.
# The ground_truth here is a simplified keyword for a PoC scoring mechanism.
EVALUATION_DATA = [
    {
        "image": "sample_image.jpg",
        "question": "Does the red forklift appear on the left side of the brown cardboard boxes?",
        "ground_truth_keyword": "left",
        "comment": "Tests relative spatial positioning.",
    },
    {
        "image": "sample_image.jpg",
        "question": "What color is the forklift?",
        "ground_truth_keyword": "red",
        "comment": "Tests basic object attribute recognition.",
    },
    {
        "image": "sample_image.jpg",
        "question": "Are there any people visible in the image?",
        "ground_truth_keyword": "no",
        "comment": "Tests for object existence.",
    },
]

# --- Core Evaluation Logic ---


def setup_environment():
    """Prepares the environment by downloading the necessary image."""
    print(f"[*] Setting up evaluation environment in {DATA_DIR}...")
    DATA_DIR.mkdir(exist_ok=True)
    if not IMAGE_PATH.exists():
        print(f"[*] Downloading sample image from {IMAGE_URL}...")
        try:
            response = requests.get(IMAGE_URL)
            response.raise_for_status()
            with open(IMAGE_PATH, "wb") as f:
                f.write(response.content)
            print(f"[+] Image saved to {IMAGE_PATH}")
        except requests.exceptions.RequestException as e:
            print(f"[!] Failed to download image: {e}")
            exit(1)


def load_model(model_id):
    """Loads the VLM and its processor."""
    print(f"[*] Loading model: {model_id}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[+] Model loaded on {DEVICE}.")
    return model, processor


def run_inference(model, processor, image_path, question):
    """
    Runs a single inference sample, inspired by the agent's 'generate' method
    in the SOURCE repository (e.g., agents/base_agent.py).
    """
    raw_image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{question} ASSISTANT:"

    inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(DEVICE)

    # Generate a response
    generate_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    # Decode the response, skipping special tokens and the prompt
    decoded_response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Extract only the assistant's part of the response
    assistant_response = decoded_response.split("ASSISTANT:")[-1].strip()
    return assistant_response


def score_answer(generated_answer, ground_truth_keyword):
    """
    A simplified scoring metric inspired by the SOURCE repo's evaluation.
    Instead of Perfect/Acceptable/Incorrect, we do a simple keyword check.
    This demonstrates the principle of automated scoring.
    """
    if ground_truth_keyword.lower() in generated_answer.lower():
        return 1, "Correct"
    return 0, "Incorrect"


def main():
    """
    Main evaluation loop, structured like local_evaluation.py from SOURCE.
    It iterates through a dataset, gets model predictions, scores them, and reports results.
    """
    setup_environment()
    model, processor = load_model(MODEL_ID)

    total_score = 0
    results = []

    print("\n" + "=" * 50)
    print("[*] Starting Evaluation Harness")
    print("=" * 50 + "\n")

    for i, item in enumerate(EVALUATION_DATA):
        print(f"--- Evaluating Sample {i+1}/{len(EVALUATION_DATA)} ---")
        question = item["question"]
        image_file = DATA_DIR / item["image"]
        ground_truth = item["ground_truth_keyword"]

        print(f"  > Question: {question}")

        generated_answer = run_inference(model, processor, image_file, question)
        print(f"  > Model Answer: {generated_answer}")

        score, status = score_answer(generated_answer, ground_truth)
        total_score += score

        print(f"  > Ground Truth Keyword: '{ground_truth}'")
        print(f"  > Result: {status} (Score: {score})")
        print(
            "-" * (len(f"--- Evaluating Sample {i+1}/{len(EVALUATION_DATA)} ---"))
            + "\n"
        )

        results.append(
            {
                "question": question,
                "generated_answer": generated_answer,
                "score": score,
                "status": status,
            }
        )

    # Final report, similar to how a benchmark would summarize performance.
    final_accuracy = (total_score / len(EVALUATION_DATA)) * 100
    print("=" * 50)
    print("[*] Evaluation Complete")
    print(f"[*] Final Score: {total_score}/{len(EVALUATION_DATA)}")
    print(f"[*] Accuracy: {final_accuracy:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
