import argparse
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# Initialize the OpenAI client, which will use the OPENAI_API_KEY environment variable.
client = OpenAI()

# The core idea from the source repo is automated testing. Here, the "test"
# is an evaluation prompt sent to a powerful "judge" LLM.
EVALUATION_PROMPT_TEMPLATE = """You are an impartial judge evaluating the quality of a generated response for a Visual Question Answering (VQA) task.
Your goal is to assess the logical consistency and factual correctness of the provided reasoning and final answer, based on the question and the description of the image content.

**Image Content Description:**
{image_description}

**Question:**
{question}

**Generated Reasoning (Chain-of-Thought):**
{reasoning}

**Generated Final Answer:**
{answer}

**Evaluation Task:**
Please evaluate the generated reasoning and answer.
1.  **Correctness:** Is the answer factually correct based on the image description?
2.  **Reasoning Quality:** Is the reasoning logical, sound, and does it lead to the final answer?
3.  **Completeness:** Does the answer fully address the question?

Provide your evaluation in a JSON object with two keys:
- "score": An integer score from 1 (very poor) to 5 (excellent).
- "justification": A brief explanation for your score, highlighting strengths and weaknesses.

**Your JSON Response:**
"""


def get_llm_evaluation(image_desc, question, reasoning, answer):
    """
    Calls a judge LLM to evaluate a VQA sample.
    """
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        image_description=image_desc,
        question=question,
        reasoning=reasoning,
        answer=answer,
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # A powerful model is needed for reliable judging
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        evaluation_str = response.choices[0].message.content
        return json.loads(evaluation_str)
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return {"score": -1, "justification": f"API Error: {str(e)}"}


def process_file(input_path, output_path):
    """
    Reads a VQA JSON, adds an evaluation to each assistant turn, and writes to a new file.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    # A simplified description of image context for the judge LLM.
    # In a real implementation, this could be more detailed (e.g., list of detected objects).
    image_description = f"An image with ID {data.get('image', 'unknown')}."

    conversations = data.get("conversations", [])
    for i in range(len(conversations)):
        if (
            conversations[i]["from"] == "assistant"
            and i > 0
            and conversations[i - 1]["from"] == "human"
        ):
            question = conversations[i - 1]["value"]
            full_response = conversations[i]["value"]

            # This logic assumes a format where reasoning precedes the final answer.
            # It may need adjustment based on the exact output of the reasoning stage.
            if "ASSISTANT:" in full_response:
                parts = full_response.split("ASSISTANT:", 1)
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = "N/A"
                answer = full_response

            evaluation = get_llm_evaluation(
                image_description, question, reasoning, answer
            )
            conversations[i]["evaluation"] = evaluation
            time.sleep(1)  # Basic rate limiting to avoid API errors

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA data using a judge LLM.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input VQA JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluated VQA JSON files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]

    print(f"Found {len(input_files)} files to process.")
    for filename in tqdm(input_files, desc="Evaluating VQA Samples"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        process_file(input_path, output_path)


if __name__ == "__main__":
    main()
