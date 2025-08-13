import os
import argparse
import openai
import json

# --- Configuration ---
# In a production environment, use a more secure way to handle API keys.
# For this experiment, we rely on an environment variable.
# `export OPENAI_API_KEY='your_api_key'`
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit(1)

# --- LLM Client (Inspired by ChatBattery's LLM_agent) ---


def ask_llm(prompt, model="gpt-4-turbo", temperature=0.7):
    """A simple wrapper for the OpenAI ChatCompletion API."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},  # For more reliable JSON output
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Fallback for models that don't support JSON mode strictly
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e_fallback:
            print(f"Fallback API call also failed: {e_fallback}")
            return None


# --- Prompt Templates (Inspired by ChatBattery's problem_conceptualization) ---

CRITIQUE_PROMPT_TEMPLATE = """
You are an expert AI creating datasets for training Vision Language Models in spatial reasoning.
Your task is to critique a given Visual Question Answering (VQA) pair based on a scene description.

**Image Scene Description:**
{scene_description}

**Initial VQA Pair:**
- Question: "{question}"
- Answer: "{answer}"

**Critique Guidelines:**
1.  **Clarity & Ambiguity:** Is the question clear? Does it refer to specific, identifiable objects?
2.  **Spatial Relevance:** Does the question genuinely test spatial reasoning (e.g., relative position, distance, orientation), or is it a simple identification task?
3.  **Triviality:** Is the question too easy or obvious? A good question should require non-trivial reasoning.

**Your Task:**
Provide a concise, constructive critique of the VQA pair. Point out its weaknesses and suggest specific areas for improvement. Respond in JSON format with a single key 'critique'.
"""

REFINEMENT_PROMPT_TEMPLATE = """
You are an expert AI creating datasets for training Vision Language Models in spatial reasoning.
Your task is to refine a VQA pair based on a provided critique to make it better for training.

**Image Scene Description:**
{scene_description}

**Initial VQA Pair to Improve:**
- Question: "{question}"
- Answer: "{answer}"

**Critique of the Initial Pair:**
{critique}

**Your Task:**
Generate a new, improved VQA pair that addresses the critique. The new question should be clearer, more spatially relevant, and less trivial. The new answer should be accurate and concise. Respond in JSON format with 'question' and 'answer' keys.
"""


def run_refinement_pipeline(scene_description, initial_question, initial_answer):
    """Runs the full generate-critique-refine pipeline."""
    print("--- [STEP 1] INITIAL VQA PAIR ---")
    print(f"  - Question: {initial_question}")
    print(f"  - Answer: {initial_answer}\n")

    # STEP 2: Critique the VQA pair using an LLM
    # This mirrors the validation step in ChatBattery (Domain Agent, Search Agent)
    print("--- [STEP 2] GENERATING CRITIQUE ---")
    critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(
        scene_description=scene_description,
        question=initial_question,
        answer=initial_answer,
    )
    critique_json_str = ask_llm(critique_prompt)
    if not critique_json_str:
        print("ERROR: Failed to generate critique.")
        return

    try:
        critique_data = json.loads(critique_json_str)
        critique = critique_data.get("critique", "No critique found in response.")
        print(f"  - LLM Critique: {critique}\n")
    except json.JSONDecodeError:
        print(
            f"ERROR: Could not parse critique JSON. Raw response:\n{critique_json_str}"
        )
        critique = critique_json_str  # Use raw string as fallback

    # STEP 3: Refine the VQA pair based on the critique
    # This mirrors the refinement prompt in ChatBattery's update logic
    print("--- [STEP 3] REFINING VQA PAIR ---")
    refinement_prompt = REFINEMENT_PROMPT_TEMPLATE.format(
        scene_description=scene_description,
        question=initial_question,
        answer=initial_answer,
        critique=critique,
    )
    refined_vqa_json_str = ask_llm(refinement_prompt)
    if not refined_vqa_json_str:
        print("ERROR: Failed to generate refined VQA pair.")
        return

    print("--- [STEP 4] FINAL REFINED VQA PAIR ---")
    try:
        # Clean up potential markdown fences
        if refined_vqa_json_str.startswith("```json"):
            refined_vqa_json_str = refined_vqa_json_str.strip("```json").strip()

        refined_vqa = json.loads(refined_vqa_json_str)
        print(f"  - Question: {refined_vqa['question']}")
        print(f"  - Answer: {refined_vqa['answer']}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: Could not parse refined VQA JSON. Error: {e}")
        print(f"  - Raw output: {refined_vqa_json_str}")


def main():
    """Main function to run the pipeline with mocked data."""
    parser = argparse.ArgumentParser(
        description="Run VQA critique and refinement pipeline."
    )
    # In a real integration, this would take an image and use vqasynth to generate
    # the scene description and initial VQA pair.
    parser.add_argument(
        "--image_path",
        type=str,
        default="assets/sample.jpg",
        help="Path to a source image (currently only used for context).",
    )
    args = parser.parse_args()

    print(
        f"INFO: Starting VQA refinement pipeline for context image: {args.image_path}\n"
    )

    # MOCKED DATA: This simulates the output of the existing VQASynth pipeline.
    scene_description = "A living room with a red sofa on the left, in front of a large window. A wooden coffee table is in front of the sofa, and a blue vase sits on the table."
    initial_question = "Is there a table?"
    initial_answer = "Yes, there is a wooden coffee table."

    run_refinement_pipeline(scene_description, initial_question, initial_answer)


if __name__ == "__main__":
    main()
