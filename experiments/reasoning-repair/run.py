import os
import json
from openai import OpenAI

# This experiment simulates the core data generation idea from APR-RL's
# `construct_buggy_dataset.py` and applies it to VQASynth's reasoning chains.
# Instead of introducing bugs into code, we introduce logical flaws into a
# model's chain of thought to create a dataset for "reasoning repair".

# --- Configuration ---
# In a real scenario, this would come from a config file.
# For this example, we assume the API key is in the environment.
# Ensure you have OPENAI_API_KEY set in your environment.
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. Using a placeholder.")
    API_KEY = "sk-placeholder"

client = OpenAI(api_key=API_KEY)

# --- Prompt inspired by APR-RL's `construct_buggy_dataset.py` ---

PROMPT_TO_INTRODUCE_FLAW = """You are an expert in spotting logical fallacies. Your task is to take a correct, well-formed reasoning chain for a Visual Question Answering (VQA) task and **introduce one realistic logical flaw**.

The flaw should:
1.  Be subtle and appear correct at first glance.
2.  Mimic a common error a VLM might make (e.g., misinterpreting spatial relations, incorrect unit conversion, a flawed calculation).
3.  Preserve the original structure and intent of the reasoning as much as possible.
4.  Not change the final question or the objects being discussed.
5.  Result in a plausible but incorrect final answer.

**Input:** A JSON object containing the `question` and a `correct_reasoning_chain`.
**Output:** Only the `flawed_reasoning_chain` as a string.

---
**EXAMPLE INPUT:**
{
    "question": "How far is the red car from the stop sign?",
    "correct_reasoning_chain": "1. The red car is identified in the left of the image. 2. The stop sign is identified in the center. 3. Depth map indicates the car is at 10 meters and the sign is at 12 meters. 4. The horizontal distance is negligible. 5. Therefore, the distance between them is approximately 2 meters. The red car is about 2 meters from the stop sign."
}

**EXAMPLE OUTPUT:**
1. The red car is identified in the left of the image. 2. The stop sign is identified in the center. 3. Depth map indicates the car is at 10 meters and the sign is at 12 meters. 4. I will add the depths to find the distance. 10 + 12 = 22 meters. 5. Therefore, the distance between them is approximately 22 meters. The red car is about 22 meters from the stop sign.
---

**YOUR TASK:**

**INPUT:**
{
    "question": "{question}",
    "correct_reasoning_chain": "{correct_reasoning_chain}"
}

**OUTPUT:**
"""

# --- Sample Data (simulating output from VQASynth pipeline) ---

CORRECT_VQA_SAMPLE = {
    "image_id": "warehouse_sample_1.jpeg",
    "question": "Does the red forklift in the warehouse appear on the left side of the brown cardboard boxes stacked on the pallet?",
    "correct_reasoning_chain": (
        "1. Identify the 'red forklift'. It is located on the left side of the image.\n"
        "2. Identify the 'brown cardboard boxes stacked on the pallet'. They are located towards the center-right of the image.\n"
        "3. Determine the relative horizontal positions. The forklift's x-coordinates are smaller than the boxes' x-coordinates.\n"
        "4. 'Left side' means having a smaller x-coordinate in the image frame.\n"
        "5. Since the forklift's position is to the left of the boxes' position, the statement is true.\n"
        "Final Answer: Yes, the red forklift appears on the left side of the brown cardboard boxes."
    )
}

def generate_flawed_reasoning(vqa_sample, model_name="gpt-4o-mini"):
    """
    Uses an LLM to introduce a flaw into a correct reasoning chain,
    mimicking the process in `construct_buggy_dataset.py`.
    """
    print(">> Generating flawed reasoning chain...")
    prompt_input = json.dumps({
        "question": vqa_sample["question"],
        "correct_reasoning_chain": vqa_sample["correct_reasoning_chain"]
    }, indent=4)

    full_prompt = PROMPT_TO_INTRODUCE_FLAW.format(
        question=vqa_sample["question"],
        correct_reasoning_chain=vqa_sample["correct_reasoning_chain"]
    )
    
    if client.api_key == "sk-placeholder":
        print(">> OpenAI client not configured. Returning a hardcoded flawed example.")
        return (
            "1. Identify the 'red forklift'. It is located on the left side of the image.\n"
            "2. Identify the 'brown cardboard boxes stacked on the pallet'. They are located towards the center-right of the image.\n"
            "3. Determine the relative horizontal positions. The forklift is on the left, the boxes are on the right. 'Left' and 'Right' are opposites.\n"
            "4. Because they are in opposite directions, the forklift cannot be on the left side of the boxes.\n"
            "5. The statement must be false.\n"
            "Final Answer: No, the red forklift does not appear on the left side of the brown cardboard boxes."
        )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=500,
            temperature=0.5,
        )
        flawed_reasoning = completion.choices[0].message.content
        return flawed_reasoning.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def create_training_pair(correct_sample, flawed_reasoning):
    """
    Formats the data into a structure suitable for Supervised Fine-Tuning (SFT).
    The goal is to train a model to "repair" the flawed reasoning.
    """
    # This input format is hypothetical, designed for an SFT trainer.
    # It provides the model with the context (image, question) and the bad
    # reasoning, and expects the correct reasoning as the output.
    input_text = (
        f"USER: You are a VQA assistant. Analyze the user's question and "
        f"the flawed reasoning provided. Your task is to correct the reasoning "
        f"and provide a valid final answer.\n\n"
        f"Image ID: {correct_sample['image_id']}\n"
        f"Question: {correct_sample['question']}\n\n"
        f"Flawed Reasoning:\n{flawed_reasoning}\n\n"
        f"ASSISTANT:"
    )
    
    output_text = correct_sample["correct_reasoning_chain"]

    return {"input": input_text, "output": output_text}


if __name__ == "__main__":
    print("--- VQASynth Reasoning Repair Experiment ---")
    print("Inspired by the data generation strategy in APR-RL.\n")

    print("Step 1: Start with a correct VQA sample from the VQASynth pipeline.")
    print("-----------------------------------------------------------------")
    print(json.dumps(CORRECT_VQA_SAMPLE, indent=2))
    print("\n")

    print("Step 2: Use an LLM to introduce a subtle flaw into the reasoning.")
    print("-----------------------------------------------------------------")
    flawed_reasoning_chain = generate_flawed_reasoning(CORRECT_VQA_SAMPLE)

    if flawed_reasoning_chain:
        print("Correct Reasoning Chain:")
        print(CORRECT_VQA_SAMPLE["correct_reasoning_chain"])
        print("\nGenerated Flawed Reasoning Chain:")
        print(flawed_reasoning_chain)
        print("\n")

        print("Step 3: Create a training pair for a 'reasoning repair' SFT task.")
        print("-----------------------------------------------------------------")
        training_pair = create_training_pair(CORRECT_VQA_SAMPLE, flawed_reasoning_chain)
        print("This JSON object represents one sample for a fine-tuning dataset:")
        print(json.dumps(training_pair, indent=2))
