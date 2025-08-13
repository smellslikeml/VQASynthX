import os
import json
from openai import OpenAI

# --- Configuration ---
# This experiment requires an OpenAI API key.
# Set the OPENAI_API_KEY environment variable before running.
try:
    client = OpenAI()
except Exception:
    print(
        "OpenAI API key not found or invalid. Please set the OPENAI_API_KEY environment variable."
    )
    client = None

# --- Mock Scene Data ---
# In a full integration, this would come from vqasynth's scene fusion pipeline.
MOCK_SCENE_DATA = {
    "description": "A small office desk.",
    "objects": [
        {"id": "obj1", "class": "laptop", "position": "center"},
        {
            "id": "obj2",
            "class": "coffee mug",
            "color": "blue",
            "position": "right of laptop",
        },
        {"id": "obj3", "class": "notebook", "position": "left of laptop"},
        {"id": "obj4", "class": "pen", "position": "on top of notebook"},
    ],
    "relationships": [
        "The coffee mug is to the right of the laptop.",
        "The notebook is to the left of the laptop.",
        "The pen is on the notebook.",
    ],
}


def generate_initial_vqa(scene_data):
    """
    Stage 1: Generate an initial VQA pair based on scene data.
    Inspired by ChatBattery's 'initial' mode in problem_conceptualization.
    """
    print("--- Stage 1: Initial VQA Generation ---")

    prompt = f"""
    You are an AI assistant creating training data for a Vision Language Model.
    Based on the following scene description, generate one question and a concise answer about the spatial relationship between objects.

    Scene Data:
    {json.dumps(scene_data, indent=2)}

    Your output must be a JSON object with two keys: "question" and "answer".
    Example: {{"question": "What is to the left of the laptop?", "answer": "A notebook."}}
    """

    if not client:
        print("OpenAI client not available. Returning mock data.")
        return {
            "question": "What is next to the laptop?",
            "answer": "A coffee mug and a notebook.",
        }

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates VQA data in JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    generated_json = json.loads(response.choices[0].message.content)
    print(f"Generated VQA: {generated_json}")
    return generated_json


def validate_vqa(vqa_pair, scene_data):
    """
    Stage 2: Validate the generated VQA pair for clarity and correctness.
    This simulates the Domain/Search agents in ChatBattery.
    """
    print("\n--- Stage 2: Validation ---")
    errors = []
    question = vqa_pair.get("question", "").lower()

    # Validation Rule 1: Check for ambiguity.
    ambiguous_terms = ["next to", "near", "close to", "by"]
    for term in ambiguous_terms:
        if f" {term} " in f" {question} ":
            error_message = f"The question '{vqa_pair['question']}' is ambiguous because it uses the vague term '{term}'. Questions should use specific directional prepositions like 'left of', 'right of', 'on top of'."
            errors.append(error_message)
            print(f"Validation FAILED: {error_message}")
            break

    if not errors:
        print("Validation PASSED: Question is sufficiently specific.")

    return errors


def refine_vqa(original_vqa, errors, scene_data):
    """
    Stage 3: Refine the VQA pair based on validation feedback.
    Inspired by ChatBattery's 'update_with_generated_battery_list' mode.
    """
    print("\n--- Stage 3: VQA Refinement ---")

    if not errors:
        print("No errors found. No refinement needed.")
        return original_vqa

    error_str = "\n".join(f"- {e}" for e in errors)
    prompt = f"""
    You are an AI assistant improving training data for a Vision Language Model.
    An initial VQA pair was generated, but it has flaws. Your task is to correct it.

    Scene Data:
    {json.dumps(scene_data, indent=2)}

    Original Flawed VQA Pair:
    {json.dumps(original_vqa, indent=2)}

    Identified Flaws:
    {error_str}

    Instructions:
    Generate a new, improved VQA pair that addresses all the identified flaws. The new question must be specific and unambiguous. The answer must be correct based on the scene data.
    Your output must be a JSON object with two keys: "question" and "answer".
    """

    if not client:
        print("OpenAI client not available. Returning mock data.")
        return {
            "question": "What object is to the right of the laptop?",
            "answer": "A blue coffee mug.",
        }

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",  # Use a more capable model for refinement
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that refines VQA data in JSON format based on feedback.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    refined_json = json.loads(response.choices[0].message.content)
    print(f"Refined VQA: {refined_json}")
    return refined_json


def main():
    """
    Orchestrates the agentic generate-validate-refine loop.
    """
    print("Starting Agentic VQA Generation Experiment...\n")

    # Stage 1: Generate
    initial_vqa = generate_initial_vqa(MOCK_SCENE_DATA)

    # Stage 2: Validate
    validation_errors = validate_vqa(initial_vqa, MOCK_SCENE_DATA)

    # Stage 3: Refine
    final_vqa = refine_vqa(initial_vqa, validation_errors, MOCK_SCENE_DATA)

    print("\n--- Experiment Summary ---")
    print(f"Initial VQA: {initial_vqa}")
    if validation_errors:
        print(
            f"Validation Feedback: {validation_errors[0]}"
        )  # a bit cleaner for the demo
        print(f"Final VQA:     {final_vqa}")
    else:
        print("Initial VQA passed validation and was accepted.")


if __name__ == "__main__":
    if client:
        main()
    else:
        print("\nExecution halted. Please set your OPENAI_API_KEY.")
