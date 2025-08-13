import json
import re

# Placeholder for vqasynth scene data derived from vision models
SCENE_DATA = {
    "image_id": "warehouse_sample_1.jpeg",
    "objects": ["red forklift", "brown cardboard boxes", "wooden pallet"],
    "description": "A warehouse scene with a red forklift to the left of stacked brown cardboard boxes.",
}


# Placeholder for an LLM call. Responses are hardcoded for testability.
def call_llm(prompt):
    """Simulates a call to a Large Language Model."""
    print(f"--- LLM PROMPT ---\n{prompt}\n--------------------")
    # If the prompt is for refinement, return a better set of VQA pairs.
    if "replace the flawed questions" in prompt.lower():
        return '- {"question": "What is the spatial orientation of the red forklift relative to the cardboard boxes?", "answer": "The red forklift is to the left of the brown cardboard boxes."}\n- {"question": "Are the wooden pallets located behind the forklift?", "answer": "The position of the wooden pallets relative to the forklift cannot be determined from the scene description."}\n- {"question": "Which object is stacked, the forklift or the boxes?", "answer": "The brown cardboard boxes are stacked."}'
    # Otherwise, return the initial, flawed set of VQA pairs.
    else:
        return '- {"question": "Is there a forklift?", "answer": "Yes, there is a red forklift."}\n- {"question": "Is there a forklift?", "answer": "Yes, there is a red forklift."}\n- {"question": "What color is the car?", "answer": "There is no car in the scene."}'


def validate_vqa_pairs(vqa_pairs, scene_objects):
    """
    Validates generated VQA pairs based on simple heuristics.
    This mimics the 'Domain Agent' and 'Search Agent' from ChatBattery.
    """
    valid_pairs = []
    flawed_questions = {"duplicates": [], "hallucinations": [], "too_simple": []}
    seen_questions = set()

    for pair in vqa_pairs:
        question = pair.get("question", "").lower()
        if not question:
            continue

        # 1. Duplication check (like ChatBattery's novelty check)
        if question in seen_questions:
            flawed_questions["duplicates"].append(pair["question"])
            continue
        seen_questions.add(question)

        # 2. Hallucination check (mentions objects not in the scene)
        if "car" in question:
            flawed_questions["hallucinations"].append(pair["question"])
            continue

        # 3. Simplicity check (ensures question requires more than simple existence check)
        if question.startswith("is there a"):
            flawed_questions["too_simple"].append(pair["question"])
            continue

        valid_pairs.append(pair)

    return valid_pairs, flawed_questions


def build_refinement_prompt(flawed_questions, scene_data):
    """
    Constructs a new prompt to ask the LLM to fix its mistakes.
    This is analogous to 'problem_conceptualization' in ChatBattery's main.py.
    """
    prompt = f"You are an expert in generating spatial reasoning questions. Based on the following scene, you previously generated flawed questions that must be replaced:\n\n"
    prompt += f"Scene Objects: {', '.join(scene_data['objects'])}\n"
    prompt += f"Scene Description: {scene_data['description']}\n\n"
    prompt += "Please generate new, high-quality questions to replace the flawed ones listed below. Ensure the new questions are complex, novel, and only reference the objects present in the scene.\n"

    if flawed_questions["duplicates"]:
        prompt += "\nThese questions were duplicates:\n- " + "\n- ".join(
            flawed_questions["duplicates"]
        )
    if flawed_questions["hallucinations"]:
        prompt += (
            "\nThese questions mentioned objects not in the scene:\n- "
            + "\n- ".join(flawed_questions["hallucinations"])
        )
    if flawed_questions["too_simple"]:
        prompt += (
            "\nThese questions were too simple and did not require spatial reasoning:\n- "
            + "\n- ".join(flawed_questions["too_simple"])
        )

    prompt += "\n\nProvide the replacements as a bulleted list of JSON objects, each with a 'question' and 'answer' key."
    return prompt


def parse_llm_output(text_output):
    """A simple parser for the LLM's bulleted list of JSON strings."""
    try:
        # Use regex to find all substrings that look like JSON objects
        json_strings = re.findall(r"\{.*?\}", text_output)
        return [json.loads(s.replace("'", '"')) for s in json_strings]
    except json.JSONDecodeError:
        print("Error decoding LLM output.")
        return []


def main():
    """Main orchestration loop demonstrating the generative feedback process."""
    print("--- STAGE 1: Initial VQA Generation ---")
    initial_prompt = f"Generate 3 VQA pairs about spatial relationships based on this scene: {SCENE_DATA['description']}. The objects are: {', '.join(SCENE_DATA['objects'])}. Return a bulleted list of JSON objects with 'question' and 'answer' keys."

    generated_text = call_llm(initial_prompt)
    initial_vqa_pairs = parse_llm_output(generated_text)
    print(f"\nInitial generated pairs: {json.dumps(initial_vqa_pairs, indent=2)}")

    print("\n--- STAGE 2: Validation --- ")
    valid_pairs, flawed_questions = validate_vqa_pairs(
        initial_vqa_pairs, SCENE_DATA["objects"]
    )
    print(f"Valid pairs from initial run: {json.dumps(valid_pairs, indent=2)}")
    print(f"Flawed questions identified: {json.dumps(flawed_questions, indent=2)}")

    num_to_regenerate = sum(len(v) for v in flawed_questions.values())

    if num_to_regenerate > 0:
        print(f"\n--- STAGE 3: Refinement ({num_to_regenerate} pairs to fix) ---")
        refinement_prompt = build_refinement_prompt(flawed_questions, SCENE_DATA)

        refined_text = call_llm(refinement_prompt)
        refined_vqa_pairs = parse_llm_output(refined_text)
        print(f"\nGenerated replacements: {json.dumps(refined_vqa_pairs, indent=2)}")

        final_vqa_dataset = valid_pairs + refined_vqa_pairs
    else:
        final_vqa_dataset = valid_pairs

    print("\n--- FINAL DATASET --- ")
    print(json.dumps(final_vqa_dataset, indent=2))


if __name__ == "__main__":
    main()
