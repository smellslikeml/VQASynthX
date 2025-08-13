import os
import json
import argparse
import pandas as pd
from openai import OpenAI

# It's assumed that the OpenAI API key is set as an environment variable.
# In a real implementation, this might use a more robust configuration system.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_critique_prompt(question, answer, reasoning):
    """
    Creates a prompt for an LLM to act as a "critic" for a VQA pair.
    """
    return f"""
You are an expert in evaluating spatial reasoning questions for vision-language models.
Your task is to critique the following VQA pair. Be concise and specific.
Identify potential flaws such as:
- Ambiguity in the question.
- Factual incorrectness in the answer (assuming the reasoning is a proxy for scene context).
- Logical fallacies or errors in the Chain-of-Thought reasoning.
- The question being too trivial or simple.

Question: "{question}"
Answer: "{answer}"
Reasoning: "{reasoning}"

Provide your critique in a single, direct sentence. If there are no obvious flaws, respond with "No major flaws detected."
Critique:
"""


def get_refinement_prompt(question, answer, reasoning, critique):
    """
    Creates a prompt for an LLM to act as a "refiner" based on a critique.
    """
    return f"""
You are an expert in generating high-quality spatial reasoning questions for vision-language models.
Your task is to refine a VQA pair based on a given critique.
The goal is to improve the quality of the data for model training.

Original Question: "{question}"
Original Answer: "{answer}"
Original Reasoning: "{reasoning}"

Critique of the original pair: "{critique}"

Please generate a new, improved VQA pair that addresses the critique.
The new question should be clear, the answer accurate, and the reasoning sound.
Output the refined VQA pair in a valid JSON format with keys "refined_question", "refined_answer", and "refined_reasoning".

Do not include any text outside of the JSON object.

{{
  "refined_question": "",
  "refined_answer": "",
  "refined_reasoning": ""
}}
"""


def process_vqa_pair(vqa_item):
    """
    Processes a single VQA pair through the critique and refinement loop.
    """
    question = vqa_item.get("question")
    answer = vqa_item.get("answer")
    reasoning = vqa_item.get("reasoning", "")  # CoT reasoning

    # 1. Get Critique
    critique_prompt = get_critique_prompt(question, answer, reasoning)
    critique_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": critique_prompt}],
        max_tokens=100,
        temperature=0.1,
    )
    critique = critique_response.choices[0].message.content.strip()

    vqa_item["critique"] = critique

    # 2. Refine if necessary
    if "no major flaws" not in critique.lower():
        refinement_prompt = get_refinement_prompt(question, answer, reasoning, critique)
        refinement_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": refinement_prompt}],
            response_format={"type": "json_object"},
            max_tokens=512,
            temperature=0.5,
        )
        try:
            refined_data = json.loads(refinement_response.choices[0].message.content)
            vqa_item.update(refined_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing refinement response: {e}")
            # Keep original data if refinement fails
            vqa_item["refined_question"] = vqa_item["question"]
            vqa_item["refined_answer"] = vqa_item["answer"]
            vqa_item["refined_reasoning"] = vqa_item["reasoning"]
    else:
        # If no flaws, the refined version is the same as the original
        vqa_item["refined_question"] = vqa_item["question"]
        vqa_item["refined_answer"] = vqa_item["answer"]
        vqa_item["refined_reasoning"] = vqa_item["reasoning"]

    return vqa_item


def main(input_file, output_file):
    """
    Main processing loop. Reads data, applies refinement, and saves output.
    """
    # Assuming input is a JSONL file with one JSON object per line.
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            try:
                vqa_item = json.loads(line)
                refined_item = process_vqa_pair(vqa_item)
                f_out.write(json.dumps(refined_item) + "\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    print(f"Refinement complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine VQA data using an LLM-based critique and refinement loop."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSONL file from the reasoning stage.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the output refined JSONL file."
    )

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    main(args.input, args.output)
