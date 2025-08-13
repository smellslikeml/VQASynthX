import os
import openai
import json

# --- Configuration ---
# In a real integration, this would use a shared client or config
# For this experiment, we'll initialize it here.
# Ensure the OPENAI_API_KEY environment variable is set.
try:
    # Using the gpt-4-turbo model is recommended for quality generation
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("Error: Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# --- Mock Scene Data ---
# This data simulates the output from previous stages in the VQASynth pipeline,
# providing context about objects detected in an image.
MOCK_SCENE_DATA = {
    "image_description": "A living room with a couch, a coffee table, and a TV.",
    "objects": [
        {
            "id": 1,
            "class": "couch",
            "color": "grey",
            "position_2d": [100, 300, 500, 450],
            "relative_position_3d": "center, far",
        },
        {
            "id": 2,
            "class": "coffee table",
            "color": "brown",
            "position_2d": [250, 480, 450, 580],
            "relative_position_3d": "center, middle",
        },
        {
            "id": 3,
            "class": "tv",
            "color": "black",
            "position_2d": [200, 100, 400, 250],
            "relative_position_3d": "center, far",
        },
        {
            "id": 4,
            "class": "cushion",
            "color": "blue",
            "position_2d": [120, 320, 180, 370],
            "relative_position_3d": "on the grey couch",
        },
        {
            "id": 5,
            "class": "cushion",
            "color": "red",
            "position_2d": [420, 320, 480, 370],
            "relative_position_3d": "on the grey couch",
        },
    ],
}

# --- Agentic Prompts (Inspired by ChatBattery's refinement loop) ---


def generate_initial_question(scene_data):
    """Agent 1: Generates an initial spatial reasoning question."""
    prompt = f"""
Given the following scene data, generate one spatial reasoning question.
The question should be answerable using only the information provided.

Scene Data:
{json.dumps(scene_data, indent=2)}

Question:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()


def critique_question(question, scene_data):
    """Agent 2: Critiques the generated question for quality and complexity."""
    prompt = f"""
You are a critic for a VQA dataset generation pipeline. Your goal is to make questions more complex and interesting for training a Vision Language Model on spatial reasoning.

Critique the following question based on the provided scene data.
Focus on these criteria:
1.  **Simplicity:** Is the question too easy (e.g., simple color identification)?
2.  **Ambiguity:** Is it clear which object(s) the question refers to?
3.  **Reasoning Depth:** Does it require single-step or multi-step reasoning? A good question requires comparing relationships or inferring context.

Scene Data:
{json.dumps(scene_data, indent=2)}

Question to Critique:
\"{question}\"

Critique and Suggestion for Improvement:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5,
    )
    return response.choices[0].message["content"].strip()


def refine_question(original_question, critique, scene_data):
    """Agent 3: Refines the question based on the critique."""
    # This mirrors ChatBattery's feedback loop, where invalid/non-novel results
    # are fed back to the LLM for replacement.
    prompt = f"""
You are a question-rewriting expert for a VQA dataset.
Your task is to rewrite an original question based on a critique to make it better for training a spatial reasoning model.

Scene Data:
{json.dumps(scene_data, indent=2)}

Original Question:
\"{original_question}\"

Critique of the Question:
\"{critique}\"

Now, rewrite the original question to address the critique. Make it more complex, less ambiguous, and require deeper spatial reasoning. The new question must still be answerable from the scene data.

Refined Question:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()


# --- Main Execution Logic ---


def main():
    """Demonstrates the full generate-critique-refine loop."""
    print("=" * 50)
    print("VQASynth Agentic QA Generation Experiment")
    print("Inspired by the refinement loop in ChatBattery")
    print("=" * 50)
    print("\n[INFO] Using mock scene data:\n")
    print(json.dumps(MOCK_SCENE_DATA, indent=2))
    print("\n" + "-" * 50 + "\n")

    # Step 1: Generate an initial question
    print("[AGENT 1] Generating initial question...")
    initial_question = generate_initial_question(MOCK_SCENE_DATA)
    print(f"  > Initial Question: {initial_question}\n")

    # Step 2: Critique the question
    print("[AGENT 2] Critiquing the question...")
    critique = critique_question(initial_question, MOCK_SCENE_DATA)
    print(f"  > Critique: {critique}\n")

    # Step 3: Refine the question based on the critique
    print("[AGENT 3] Refining the question based on critique...")
    refined_question = refine_question(initial_question, critique, MOCK_SCENE_DATA)
    print(f"  > Refined Question: {refined_question}\n")

    print("=" * 50)
    print("Experiment Complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
