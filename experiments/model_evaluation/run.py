import argparse
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict

# --- Model Loading ---
# Cache models to avoid reloading them for each step if they are the same
MODEL_CACHE = {}
PROCESSOR_CACHE = {}


def get_model_and_processor(model_id: str, device: str):
    """Loads a VLM and its processor, caching them for reuse."""
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id], PROCESSOR_CACHE[model_id]

    print(f"Loading model: {model_id}...")
    # Trust remote code for models like Idefics2 that require it.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    MODEL_CACHE[model_id] = model
    PROCESSOR_CACHE[model_id] = processor
    return model, processor


# --- Core Logic Functions ---


def run_inference(
    model_id: str, prompt_text: str, image: Image.Image, device: str
) -> str:
    """Generic function to run inference with a LLaVA-style VLM."""
    model, processor = get_model_and_processor(model_id, device)

    # Use a standard VLM prompt format. For LLaVA, this is <image>\nUSER: {prompt}\nASSISTANT:
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generate_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # The output includes the prompt, so we decode the full string and slice it
    full_output = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Find the start of the assistant's response
    assistant_token = "ASSISTANT:"
    assistant_pos = full_output.rfind(assistant_token)
    if assistant_pos != -1:
        return full_output[assistant_pos + len(assistant_token) :].strip()
    else:
        # Fallback if the format is unexpected
        return full_output


def generate_question(model_id: str, image: Image.Image, device: str) -> str:
    """Step 1: Generate a spatial reasoning question based on the image."""
    print("Step 1: Generating question...")
    prompt = "Based on the image, generate one complex spatial reasoning question that requires estimating distance, position, or orientation between two distinct objects. Do not answer the question, only provide the question itself."
    question = run_inference(model_id, prompt, image, device)
    print(f"Generated Question: {question}")
    return question


def get_answer(model_id: str, image: Image.Image, question: str, device: str) -> str:
    """Step 2: Answer the generated question."""
    print("Step 2: Answering question...")
    # The prompt for the answerer is simply the question.
    answer = run_inference(model_id, question, image, device)
    print(f"Generated Answer: {answer}")
    return answer


def evaluate_answer(
    model_id: str, image: Image.Image, question: str, answer: str, device: str
) -> Dict:
    """Step 3: Evaluate the answer for correctness and reasoning."""
    print("Step 3: Evaluating answer...")
    prompt = f"""You are a meticulous evaluator. Your task is to assess an answer to a spatial reasoning question based on the provided image.\n\nQuestion: \"{question}\"\nAnswer to Evaluate: \"{answer}\"\n\nAnalyze the answer for spatial accuracy, logical reasoning, and correctness based on the visual evidence. Provide your evaluation in a JSON format with two keys:\n1. \"score\": An integer score from 1 (poor) to 5 (excellent).\n2. \"rationale\": A brief explanation for your score, commenting on the answer's strengths and weaknesses.\n\nDo not add any text outside the JSON structure."""
    evaluation_str = run_inference(model_id, prompt, image, device)
    print(f"Generated Evaluation: {evaluation_str}")
    try:
        # Clean up potential markdown fences and extract JSON
        if "```json" in evaluation_str:
            evaluation_str = evaluation_str.split("```json")[1].split("```")[0]
        evaluation_json = json.loads(evaluation_str)
        return evaluation_json
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing evaluation JSON: {e}")
        return {
            "score": -1,
            "rationale": f"Failed to parse evaluator output: {evaluation_str}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run a 3-step model-based evaluation for spatial reasoning, inspired by LLM-Crowdsourced."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="VLM to generate the question. Can be a fine-tuned model from VQASynth.",
    )
    parser.add_argument(
        "--answerer_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="VLM to answer the question.",
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="VLM to evaluate the answer.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load image
    try:
        image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {args.image_path}")
        return

    # --- Run the LLM-Crowdsourced inspired pipeline ---
    # 1. Generate Question
    question = generate_question(args.generator_model, image, device)

    # 2. Get Answer
    answer = get_answer(args.answerer_model, image, question, device)

    # 3. Evaluate Answer
    evaluation = evaluate_answer(args.evaluator_model, image, question, answer, device)

    # --- Final Output ---
    result = {
        "image_path": args.image_path,
        "models": {
            "generator": args.generator_model,
            "answerer": args.answerer_model,
            "evaluator": args.evaluator_model,
        },
        "evaluation_data": {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
        },
    }

    print("\n--- Evaluation Complete ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
