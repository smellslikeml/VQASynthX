import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm

# This script is inspired by the data synthesis approach in:
# https://github.com/Tomsawyerhu/APR-RL/blob/main/construct_buggy_dataset.py
# It adapts the idea of using a powerful LLM to inject realistic errors
# from the domain of code repair to spatial reasoning for VQA.


# --- LLM Client Configuration ---
def get_openai_client():
    """Initializes and returns the OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key, base_url=base_url)


# --- Prompt Template for Flawed Reasoning Synthesis ---
FLAWED_REASONING_PROMPT = """You are an expert in spatial reasoning and visual analysis. Your task is to take a correct, well-reasoned answer to a visual question and **introduce one or more realistic logical errors** into the reasoning process. The errors should:

1. **Cause the final answer to be incorrect.**
2. **Be logically misleading** — they should appear correct at first glance but contain subtle flaws (e.g., misinterpreting a spatial relationship, making a small calculation error, confusing left/right, misjudging distance).
3. **Mimic common human cognitive biases** or simple mistakes.
4. **Preserve the original structure and tone** of the reasoning as much as possible.
5. **Do not change the question.**
6. Respond ONLY with the flawed reasoning text, starting with the reasoning steps and ending with the final answer.

Here is an example.
Input:
[Start of Correct Reasoning]
Question: Is the red fire hydrant to the left of the silver car?
Reasoning:
1. Identify the red fire hydrant in the image.
2. Identify the silver car in the image.
3. From the camera's perspective, the fire hydrant is located on the right side of the frame, and the car is located on the left side.
4. Therefore, the fire hydrant is to the right of the car, not the left.
Final Answer: No.
[End of Correct Reasoning]

Output:
[Start of Flawed Reasoning]
Reasoning:
1. Identify the red fire hydrant in theimage.
2. Identify the silver car in the image.
3. From the camera's perspective, the fire hydrant is closer to the viewer than the car.
4. The hydrant is on the left side of the street, and the car is on the right. (\u274c Flaw: Confuses the object's position relative to the street with its position relative to the other object from the camera's perspective).
5. Therefore, the fire hydrant is to the left of the car.
Final Answer: Yes.
[End of Flawed Reasoning]

Now, you are given a correct reasoning chain. Return a flawed version.
Input:
[Start of Correct Reasoning]
{correct_reasoning}
[End of Correct Reasoning]

Output:
[Start of Flawed Reasoning]
"""


def generate_flawed_reasoning(client, correct_reasoning, model_name="gpt-4o-mini"):
    """Generates a flawed version of the reasoning using an LLM."""
    try:
        prompt = FLAWED_REASONING_PROMPT.format(correct_reasoning=correct_reasoning)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.5,
            n=1,
        )
        if completion.choices:
            response = completion.choices[0].message.content.strip()
            # Clean up potential markdown fences from the model output
            if response.endswith("[End of Flawed Reasoning]"):
                response = response.replace("[End of Flawed Reasoning]", "").strip()
            return response
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
    return None


def process_vqa_data(input_file, output_file):
    """Reads VQA data, generates flawed counterparts, and writes to a new file."""
    client = get_openai_client()
    processed_samples = []

    with open(input_file, "r") as f_in:
        samples = [json.loads(line) for line in f_in]

    for sample in tqdm(samples, desc="Generating Flawed Reasoning"):
        # Assuming the input format has 'question' and 'correct_reasoning' fields
        if (
            "question" not in sample
            or "reasoning" not in sample
            or "answer" not in sample
        ):
            continue

        correct_reasoning_text = (
            f"Question: {sample['question']}\n"
            f"Reasoning:\n{sample['reasoning']}\n"
            f"Final Answer: {sample['answer']}"
        )

        flawed_output = generate_flawed_reasoning(client, correct_reasoning_text)

        if flawed_output:
            # Create a new sample with original and flawed data for DPO
            dpo_sample = {
                "image_id": sample.get("image_id"),
                "question": sample["question"],
                "chosen": f"{sample['reasoning']}\nFinal Answer: {sample['answer']}",
                "rejected": flawed_output,
            }
            processed_samples.append(dpo_sample)

    with open(output_file, "w") as f_out:
        for sample in processed_samples:
            f_out.write(json.dumps(sample) + "\n")

    print(
        f"Processing complete. Wrote {len(processed_samples)} DPO-ready samples to {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate flawed reasoning for VQA data."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input JSONL file with correct VQA reasoning.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output JSONL file for DPO training.",
    )
    args = parser.parse_args()

    process_vqa_data(args.input_file, args.output_file)
