import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm

# Initialize OpenAI client
# Assumes OPENAI_API_KEY is set in the environment
try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure the OPENAI_API_KEY environment variable is set.")
    exit(1)

# The core prompt for the LLM-as-a-Judge
# Inspired by the risk-aware prompts from MIRAGE-Bench
VERIFICATION_PROMPT_TEMPLATE = """
You are an expert verifier for a Visual Question Answering (VQA) dataset. Your task is to evaluate the quality and correctness of a synthetically generated question-answer pair based on a description of an image scene.

**Critically evaluate the provided Answer based on the Question and the Scene Description.**

**Scene Description:**
{scene_description}

**Question:**
{question}

**Generated Answer:**
{answer}

**Evaluation Criteria:**
1.  **Factual Consistency:** Is the answer factually consistent with the objects, locations, and relationships described in the Scene Description?
2.  **Logical Soundness:** Does the answer follow logically from the question? For reasoning (Chain-of-Thought) answers, is the reasoning process sound?
3.  **Spatial Accuracy:** Does the answer correctly interpret spatial relationships (e.g., "left of", "behind", "closer to") and distances?

**Your Task:**
Respond with a JSON object containing two keys:
- "judgment": A single string, either "PASS" or "FAIL". "PASS" if the answer is correct and high-quality. "FAIL" if it is factually incorrect, logically flawed, or spatially inaccurate.
- "reasoning": A brief, one-sentence explanation for your judgment.

**Example Response:**
{{"judgment": "FAIL", "reasoning": "The answer incorrectly states the forklift is behind the boxes, but the scene description places it to the left."}}

Now, provide your evaluation for the given data.

**JSON Response:**
"""


def verify_vqa_pair(vqa_item):
    """
    Uses an LLM to verify a single VQA pair.
    """
    # In a real VQASynth pipeline, the scene_description would be constructed
    # from the structured scene graph data. For this experiment, we assume
    # it's a pre-existing field in the input data.
    scene_description = vqa_item.get(
        "scene_description", "No scene description provided."
    )
    question = vqa_item.get("question")
    answer = vqa_item.get("answer")

    if not all([question, answer]):
        return {"error": "Missing question or answer in the input item."}

    prompt = VERIFICATION_PROMPT_TEMPLATE.format(
        scene_description=scene_description, question=question, answer=answer
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {
            "error": f"API call failed: {e}",
            "judgment": "ERROR",
            "reasoning": str(e),
        }


def create_sample_data(filepath):
    """Creates a dummy input file for demonstration purposes."""
    sample_data = [
        {
            "id": "sample_1",
            "scene_description": "A red forklift is positioned to the left of a tall stack of brown cardboard boxes. A man in a red hat is standing approximately 5 meters away from the boxes.",
            "question": "Does the red forklift in the warehouse appear on the left side of the brown cardboard boxes stacked?",
            "answer": "Yes, the red forklift is on the left side of the stacked brown cardboard boxes.",
        },
        {
            "id": "sample_2",
            "scene_description": "A red forklift is positioned to the left of a tall stack of brown cardboard boxes. A man in a red hat is standing approximately 5 meters away from the boxes.",
            "question": "How far is the forklift from the man in the red hat?",
            "answer": "The forklift is 2 meters away from the man.",
        },
        {
            "id": "sample_3",
            "scene_description": "A green chair is placed directly in front of a wooden desk. A blue lamp is on the right side of the desk.",
            "question": "What object is behind the green chair?",
            "answer": "The wooden desk is behind the green chair.",
        },
        {
            "id": "sample_4",
            "scene_description": "A green chair is placed directly in front of a wooden desk. A blue lamp is on the right side of the desk.",
            "question": "Is the blue lamp on the right side of the green chair?",
            "answer": "Yes, the blue lamp is on the right side of the desk, which means it is also to the right of the chair.",
        },
    ]
    with open(filepath, "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    print(f"Created sample data file at {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify VQA data using an LLM-as-a-Judge."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="vqa_data.jsonl",
        help="Path to the input JSONL file with VQA data.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="verification_results.jsonl",
        help="Path to save the verification results.",
    )
    args = parser.parse_args()

    # Create a sample input file if it doesn't exist
    if not os.path.exists(args.input_file):
        print(f"Input file not found. Creating a sample file: {args.input_file}")
        create_sample_data(args.input_file)

    with open(args.input_file, "r") as f_in, open(args.output_file, "w") as f_out:
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Verifying VQA pairs"):
            try:
                vqa_item = json.loads(line)
                result = verify_vqa_pair(vqa_item)

                output_record = {"original_data": vqa_item, "verification": result}

                f_out.write(json.dumps(output_record) + "\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    print(f"Verification complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
