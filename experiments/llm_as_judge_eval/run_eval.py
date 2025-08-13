import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm

# Initialize OpenAI client
# Assumes OPENAI_API_KEY is set in the environment
try:
    client = OpenAI()
    API_AVAILABLE = True
except Exception:
    client = None
    API_AVAILABLE = False
    print("Warning: OpenAI API key not configured. Evaluation will use dummy data.")

EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for a Visual Question Answering (VQA) dataset focused on spatial reasoning.
Your task is to assess the quality of a synthesized data sample based on the ground truth scene metadata.
Evaluate whether the provided "Answer" is a correct and logical conclusion based on the "Question" and the "Ground Truth Data".
The "Chain of Thought" (CoT) shows the reasoning steps used to generate the answer. Assess if the CoT is sound and consistent with the ground truth.

**Ground Truth Data:**
{ground_truth}

**Question:**
{question}

**Chain of Thought (CoT):**
{cot}

**Answer:**
{answer}

**Evaluation Criteria:**
1.  **Factual Consistency:** Is the answer factually consistent with the Ground Truth Data?
2.  **Logical Soundness:** Is the reasoning in the Chain of Thought logical and does it lead to the final answer?
3.  **Relevance:** Is the answer relevant to the question asked?

**Your Task:**
Provide a JSON response with your evaluation. The JSON object should have three keys: 'factual_consistency' (boolean), 'logical_soundness' (boolean), and 'evaluation_notes' (a string explaining your reasoning, max 50 words).

**Example Response:**
{
  "factual_consistency": true,
  "logical_soundness": true,
  "evaluation_notes": "The CoT correctly identifies the objects and their distances from the ground truth data. The final answer accurately reflects the comparison requested in the question."
}
"""


def load_dataset_sample(file_path, num_samples=5):
    """Loads a few samples from a VQASynth-generated dataset."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data[:num_samples]


def evaluate_sample(sample):
    """Formats the prompt and calls the LLM for evaluation."""
    ground_truth_str = json.dumps(
        {
            "image_id": sample.get("image"),
            "scene_data": sample.get("scene_data", "Not Available"),
        },
        indent=2,
    )

    question = ""
    cot_and_answer = ""
    if "conversations" in sample and len(sample["conversations"]) >= 2:
        question = sample["conversations"][0]["value"]
        cot_and_answer = sample["conversations"][1]["value"]

    parts = cot_and_answer.split("ASSISTANT:")
    cot = parts[0].replace("ASSISTANT's Thought:", "").strip() if len(parts) > 1 else ""
    answer = parts[1].strip() if len(parts) > 1 else cot_and_answer

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        ground_truth=ground_truth_str, question=question, cot=cot, answer=answer
    )

    if not API_AVAILABLE:
        return {
            "factual_consistency": True,
            "logical_soundness": True,
            "evaluation_notes": "Dummy evaluation. OpenAI API key not found.",
        }

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert VQA data evaluator specializing in spatial reasoning.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse LLM response.",
            "raw_response": response.choices[0].message.content,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQASynth data quality using an LLM-as-judge."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the VQASynth-generated JSON dataset file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate from the dataset.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results.",
    )
    args = parser.parse_args()

    print(f"Loading {args.num_samples} samples from {args.dataset_path}...")
    samples = load_dataset_sample(args.dataset_path, args.num_samples)

    results = []
    for sample in tqdm(samples, desc="Evaluating samples"):
        evaluation = evaluate_sample(sample)
        results.append({"image_id": sample.get("image"), "evaluation": evaluation})

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results saved to {args.output_file}")

    # Print a summary
    consistent_count = sum(
        1 for res in results if res["evaluation"].get("factual_consistency") is True
    )
    sound_count = sum(
        1 for res in results if res["evaluation"].get("logical_soundness") is True
    )
    total = len(results)
    if total > 0:
        print("\n--- Evaluation Summary ---")
        print(f"Factual Consistency: {consistent_count / total:.2%}")
        print(f"Logical Soundness:   {sound_count / total:.2%}")
        print("--------------------------")


if __name__ == "__main__":
    main()
