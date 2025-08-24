import os
import requests
import json
import base64
from io import BytesIO

# --- Configuration ---
# Fetched from environment variables, similar to SOURCE repo's Docker setup
API_KEY = os.environ.get("API_KEY")
API_BASE = os.environ.get(
    "API_BASE", "https://api.openai.com/v1"
)  # Default for OpenAI compatible APIs
MODEL_A = os.environ.get("MODEL_A", "gpt-4o")
MODEL_B = os.environ.get("MODEL_B", "llava-hf/llava-1.5-7b-hf")  # Example placeholder

# A sample image for VQA evaluation
SAMPLE_IMAGE_URL = "https://github.com/smellslikeml/experimental-vqasynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"

# --- Prompts inspired by LLM-Crowdsourced's methodology ---

QUESTION_GENERATION_PROMPT = """
You are an expert in spatial reasoning and visual question answering.
Based on the provided image, generate one insightful and non-trivial question that tests spatial reasoning.
The question should inquire about the relative position, orientation, or distance between two distinct objects in the scene.
Output your question in a JSON object with a single key \"question\".

Example:
{
  "question": "Is the red forklift located to the left or right of the stack of brown cardboard boxes from the camera's perspective?"
}
"""

EVALUATION_PROMPT_TEMPLATE = """
You are an impartial judge evaluating the quality of a Vision Language Model's answer to a spatial reasoning question about an image.
Your task is to assess the provided answer for correctness, clarity, and reasoning.

**Image Context:** The evaluation is based on the provided image.
**Question:** "{question}"
**Answer to Evaluate:** "{answer}"

**Evaluation Criteria:**
1.  **Correctness:** Is the answer factually correct based on the visual evidence?
2.  **Completeness:** Does the answer fully address the question?
3.  **Reasoning:** Does the answer explain *why* it is correct, referencing objects in the scene?

**Your Task:**
Provide a score from 1 to 5 (1=Poor, 5=Excellent) and a brief justification for your score.
Output your evaluation in a JSON object with two keys: "score" (int) and "justification" (string).
"""

# --- API Interaction Logic (Simplified from SOURCE's models.py) ---


def get_image_b64(image_url: str) -> str:
    """Fetches an image from a URL and returns it as a base64 encoded string."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_bytes = BytesIO(response.content)
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None


def call_vlm_api(model: str, prompt: str, image_b64: str) -> dict:
    """A generic function to call a VLM API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3,
    }

    # Use API_BASE to allow for different backends (OpenAI, Anyscale, etc.)
    endpoint = f"{API_BASE}/chat/completions"

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Try to parse as JSON, otherwise return the raw string
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}  # Return as dict for consistency
    except requests.exceptions.RequestException as e:
        print(f"API call failed for model {model}: {e}")
        return {"error": str(e)}


# --- Main Experiment Logic (Inspired by SOURCE's math_experiment.py) ---


def run_mutual_evaluation_round(
    questioner_model: str, answerer_model: str, image_b64: str
):
    """
    Runs a single round of evaluation:
    1. Questioner model generates a question.
    2. Answerer model answers it.
    3. Questioner model evaluates the answer.
    """
    print(
        f"\n--- Starting Round: [{questioner_model}] asks, [{answerer_model}] answers ---"
    )

    # 1. Generate Question
    print(f"[{questioner_model}] is generating a question...")
    q_data = call_vlm_api(questioner_model, QUESTION_GENERATION_PROMPT, image_b64)
    if "error" in q_data or "question" not in q_data:
        print("Failed to generate a valid question. Aborting round.")
        return
    question = q_data["question"]
    print(f"Generated Question: {question}")

    # 2. Generate Answer
    print(f"[{answerer_model}] is answering the question...")
    a_data = call_vlm_api(answerer_model, question, image_b64)
    if "error" in a_data:
        print("Failed to generate an answer. Aborting round.")
        return
    answer = a_data.get("text", str(a_data))  # Handle non-JSON response
    print(f"Generated Answer: {answer}")

    # 3. Evaluate Answer
    print(f"[{questioner_model}] is evaluating the answer...")
    eval_prompt = EVALUATION_PROMPT_TEMPLATE.format(question=question, answer=answer)
    eval_data = call_vlm_api(questioner_model, eval_prompt, image_b64)
    if "error" in eval_data:
        print("Failed to generate an evaluation. Aborting round.")
        return
    print(
        f"Evaluation: Score={eval_data.get('score', 'N/A')}, Justification='{eval_data.get('justification', 'N/A')}'"
    )
    print("--- Round Complete ---")


def main():
    """Main function to run the VQA mutual evaluation experiment."""
    if not API_KEY:
        raise ValueError("API_KEY environment variable is not set.")

    print("Starting VQA Mutual Evaluation Experiment...")
    print(f"Model A (e.g., Questioner/Evaluator): {MODEL_A}")
    print(f"Model B (e.g., Answerer): {MODEL_B}")

    # Load and prepare the image
    print(f"Fetching sample image from: {SAMPLE_IMAGE_URL}")
    image_b64 = get_image_b64(SAMPLE_IMAGE_URL)
    if not image_b64:
        print("Could not load image. Exiting.")
        return

    # Run two rounds for symmetry, like in LLM-Crowdsourced
    run_mutual_evaluation_round(
        questioner_model=MODEL_A, answerer_model=MODEL_B, image_b64=image_b64
    )
    run_mutual_evaluation_round(
        questioner_model=MODEL_B, answerer_model=MODEL_A, image_b64=image_b64
    )


if __name__ == "__main__":
    main()
