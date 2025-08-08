import argparse
import os
import json
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- EESE Inspired Judging Logic ---

def get_judge_score(client, judge_model: str, question: str, ground_truth: str, model_answer: str) -> int:
    """Uses a judge LLM to score the model's answer against the ground truth."""
    prompt = f"""Please evaluate the following model-generated answer based on the provided question and ground truth reference. Your evaluation should focus on correctness, relevance, and accuracy.

- Question: {question}
- Ground Truth Answer: {ground_truth}
- Model's Answer: {model_answer}

This is a spatial reasoning question. Please score the answer on a scale of 0 to 10, where 10 is perfectly correct and 0 is completely incorrect. Provide only an integer score and no other text or explanation."""

    messages = [
        {"role": "system", "content": "You are an impartial evaluator for an AI model's spatial reasoning capabilities."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=0.0,
            max_tokens=5
        )
        score_text = response.choices[0].message.content.strip()
        # Extract the first numeric value found
        import re
        match = re.search(r'\d+', score_text)
        if match:
            score = int(match.group())
            return max(0, min(10, score)) # Clamp score between 0 and 10
        else:
            logging.warning(f"Could not parse score from judge's response: '{score_text}'")
            return -1 # Indicates a parsing error
    except Exception as e:
        logging.error(f"Error calling judge model: {e}")
        return -1

# --- VQASynth Model Inference ---

def get_model_prediction(model, processor, image: Image.Image, question: str, device: str) -> str:
    """Generates a prediction from a VQASynth-trained Vision Language Model."""
    prompt = f"<|user|>\n<image>\n{question}<|end|><|assistant|>"
    
    try:
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Post-process to get only the assistant's response
        # Expected format: <|user|>\n<image>\n{question}<|end|><|assistant|>\n{response}
        response = generated_texts[0].split('<|assistant|>')[1].strip()
        return response
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        return "Error generating prediction."


def main(args):
    # Initialize OpenAI client for the judge model
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        judge_client = OpenAI(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return

    # Load the model and processor to be evaluated
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model {args.model_id} on {device}...")
    try:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        model_under_test = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            token=hf_token
        ).to(device)
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=hf_token)
    except Exception as e:
        logging.error(f"Failed to load model from Hugging Face: {e}")
        return

    # Load evaluation dataset
    logging.info(f"Loading dataset {args.dataset_id}...")
    try:
        dataset = load_dataset(args.dataset_id, split=f"train[:{args.num_samples}]")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Process dataset and write results
    with open(args.output_file, 'w') as f:
        for i, item in enumerate(dataset):
            logging.info(f"Processing item {i+1}/{args.num_samples}...")
            image_url = item['image']
            question = item['question']
            ground_truth = item['answer']

            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                logging.error(f"Failed to load image from {image_url}: {e}")
                continue

            model_answer = get_model_prediction(model_under_test, processor, image, question, device)
            logging.info(f"  Question: {question}")
            logging.info(f"  GT Answer: {ground_truth}")
            logging.info(f"  Model Answer: {model_answer}")

            score = get_judge_score(judge_client, args.judge_model, question, ground_truth, model_answer)
            logging.info(f"  Assigned Score: {score}")

            result = {
                "id": i,
                "question": question,
                "ground_truth_answer": ground_truth,
                "model_answer": model_answer,
                "score": score
            }
            f.write(json.dumps(result) + '\n')

    logging.info(f"Evaluation complete. Results saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a VLM using an LLM as a judge.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID of the VLM to evaluate.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model ID for the judge LLM (e.g., gpt-4o).")
    parser.add_argument("--dataset_id", type=str, required=True, help="Hugging Face dataset ID for evaluation.")
    parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl", help="File to save evaluation results.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples from the dataset to evaluate.")
    
    args = parser.parse_args()
    main(args)
