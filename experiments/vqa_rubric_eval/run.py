# experiments/vqa_rubric_eval/run.py

import os
import json
import argparse
from openai import OpenAI
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize OpenAI client
# Ensure the OPENAI_API_KEY environment variable is set.
try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    logging.error(
        "The 'OPENAI_API_KEY' environment variable is not set. Please set it before running the script."
    )
    exit(1)

EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for Vision-Question-Answering (VQA) datasets.
Your task is to assess the quality of a synthetically generated VQA sample based on its textual components.

**Evaluation Criteria:**

1.  **Question Plausibility (1-5):** How plausible and clear is the question?
    - 1: Nonsensical, grammatically incorrect, or completely ambiguous.
    - 3: Understandable but could be phrased better or is overly simplistic.
    - 5: Clear, specific, and relevant to a typical spatial reasoning scenario.

2.  **Answer Correctness & Coherence (1-5):** Assuming the (unseen) image context, how correct and coherent is the answer, especially the chain-of-thought reasoning?
    - 1: The answer is logically flawed, self-contradictory, or the reasoning is nonsensical.
    - 3: The reasoning is understandable but contains minor logical gaps or inaccuracies.
    - 5: The answer is well-reasoned, coherent, and follows a logical chain of thought.

3.  **Spatial Reasoning Complexity (1-5):** How well does the QA pair capture complex spatial reasoning?
    - 1: Trivial reasoning (e.g., simple object presence).
    - 3: Involves basic relationships (e.g., left/right of, near/far).
    - 5: Involves complex, multi-step reasoning, distance estimation, or orientation.

**VQA Sample:**
---
**Question:**
{question}

**Answer (with Chain-of-Thought):**
{answer}
---

**Your Task:**
Provide a structured JSON object with your evaluation. Do not include any other text or markdown fences.

**JSON Output Format:**
{{
  "scores": {{
    "question_plausibility": <score_1_to_5>,
    "answer_coherence": <score_1_to_5>,
    "spatial_complexity": <score_1_to_5>
  }},
  "justification": "<briefly explain your reasoning for the scores>"
}}
"""


def load_vqa_data(filepath: str) -> List[Dict[str, Any]]:
    """Loads VQA data from a JSON file."""
    try:
        with open(filepath, "r") as f:
            # VQASynth datasets are often JSONL
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logging.error(
            f"Error decoding JSON from {filepath}. Ensure it is a valid JSON or JSONL file."
        )
        return []


def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates a single VQA sample using an LLM."""
    # Assuming LLaVA conversational format where 'conversations' is a list of turns
    try:
        question = sample.get("conversations", [{}])[0].get("value", "N/A")
        answer = sample.get("conversations", [{}, {}])[1].get("value", "N/A")
    except IndexError:
        logging.warning(
            f"Skipping sample with incomplete conversation: {sample.get('id', 'Unknown ID')}"
        )
        return None

    if question == "N/A" or answer == "N/A":
        logging.warning(
            f"Skipping sample with missing question or answer: {sample.get('id', 'Unknown ID')}"
        )
        return None

    try:
        # The question often includes a placeholder like <image>\n, which we can strip.
        question_text = question.split("\n")[-1].strip()
        answer_text = answer.strip()
    except (IndexError, AttributeError):
        logging.warning(
            f"Could not parse question/answer for sample: {sample.get('id', 'Unknown ID')}"
        )
        return None

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question_text, answer=answer_text
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        eval_result = json.loads(response.choices[0].message.content)
        return eval_result
    except Exception as e:
        logging.error(f"Error calling OpenAI API or parsing response: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQA-Synth dataset quality using an LLM-based rubric."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the VQA dataset file (JSONL format).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save the detailed evaluation results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to evaluate from the input file.",
    )
    args = parser.parse_args()

    vqa_data = load_vqa_data(args.input_file)
    if not vqa_data:
        return

    data_to_evaluate = vqa_data[: args.limit]
    logging.info(
        f"Loaded {len(vqa_data)} samples. Evaluating the first {len(data_to_evaluate)}."
    )

    all_results = []
    total_scores = {
        "question_plausibility": 0,
        "answer_coherence": 0,
        "spatial_complexity": 0,
    }
    valid_evals = 0

    for i, sample in enumerate(data_to_evaluate):
        logging.info(f"Evaluating sample {i+1}/{len(data_to_evaluate)}...")
        result = evaluate_sample(sample)
        if result and "scores" in result:
            all_results.append(
                {"id": sample.get("id", f"sample_{i}"), "evaluation": result}
            )
            total_scores["question_plausibility"] += result["scores"][
                "question_plausibility"
            ]
            total_scores["answer_coherence"] += result["scores"]["answer_coherence"]
            total_scores["spatial_complexity"] += result["scores"]["spatial_complexity"]
            valid_evals += 1
        else:
            logging.warning(f"Failed to evaluate sample {i+1}.")

    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Saved detailed results to {args.output_file}")

    if valid_evals > 0:
        avg_scores = {k: v / valid_evals for k, v in total_scores.items()}
        print("\n--- Evaluation Summary ---")
        print(f"Total samples evaluated: {valid_evals}")
        print("\nAverage Scores:")
        for key, val in avg_scores.items():
            print(f"  - {key.replace('_', ' ').title()}: {val:.2f}")
        print("--------------------------")
    else:
        print("No samples were successfully evaluated.")


if __name__ == "__main__":
    # For demonstration, create a dummy input file if it doesn't exist.
    DUMMY_FILE = "dummy_vqa_data.jsonl"
    if not os.path.exists(DUMMY_FILE):
        logging.info(f"Creating a dummy input file: {DUMMY_FILE}")
        dummy_data = [
            {
                "id": "sample_1",
                "image": "00001.jpg",
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nIs the red box to the left of the blue sphere?",
                    },
                    {
                        "from": "gpt",
                        "value": "THINK: The user wants to know the relative position of two objects. I need to identify the red box and the blue sphere in the image and determine their horizontal relationship. The red box is at coordinates (100, 150) and the blue sphere is at (300, 150). Since 100 < 300, the red box is to the left. Yes, the red box is to the left of the blue sphere.",
                    },
                ],
            },
            {
                "id": "sample_2",
                "image": "00002.jpg",
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nHow far is the chair from the table in meters?",
                    },
                    {
                        "from": "gpt",
                        "value": "THINK: The user is asking for a distance estimation. I need to find the chair and the table. The chair appears to be very close to the table. Based on the estimated depth map, the distance is approximately 0.5 meters. The chair is about 0.5 meters from the table.",
                    },
                ],
            },
        ]
        with open(DUMMY_FILE, "w") as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
        print(
            f"\nTo run a demo, first ensure OPENAI_API_KEY is set, then use the command:\npython {__file__} --input_file {DUMMY_FILE} --limit 2\n"
        )

    main()
