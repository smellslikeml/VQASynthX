import os


# This function simulates an LLM call.
# In a real scenario, this would be a proper API call to a VLM/LLM.
def mock_llm_call(prompt):
    print("--- LLM PROMPT ---")
    print(prompt)
    print("------------------")

    # Simulate different responses based on the prompt content
    if "initial question" in prompt:
        return {
            "question": "Is the red chair to the right of the blue sofa?",
            "answer": "Yes, from the camera's viewpoint, the red chair is positioned to the right of the blue sofa. The distance between them is approximately 2 meters. The red chair is to the left of the blue sofa.",  # Intentionally flawed
        }
    elif "has the following issue" in prompt:
        return {
            "question": "Is the red chair to the right of the blue sofa?",
            "answer": "Yes, from the camera's viewpoint, the red chair is positioned to the right of the blue sofa, approximately 2 meters away.",  # Corrected answer
        }
    else:
        return {"question": "N/A", "answer": "N/A"}


# This function simulates validating the generated answer against scene data or logical rules.
# Inspired by the Domain_Agent and Search_Agent in ChatBattery.
def validate_vqa_pair(vqa_pair, scene_data):
    """
    Validates the VQA pair. Returns a list of critiques.
    An empty list means the pair is valid.
    """
    critiques = []
    answer = vqa_pair.get("answer", "").lower()

    # Rule 1: Check for self-contradiction
    if "left of" in answer and "right of" in answer:
        critiques.append(
            "The answer is self-contradictory, mentioning both 'left of' and 'right of' for the same relationship."
        )

    # Rule 2: (Placeholder) Check against scene data
    # if not check_distance_in_scene(scene_data, "red chair", "blue sofa", "2 meters"):
    #     critiques.append("The estimated distance of '2 meters' is inconsistent with the scene geometry.")

    return critiques


# Inspired by the `problem_conceptualization` function in ChatBattery's main.py
def build_refinement_prompt(original_vqa, critiques):
    """
    Builds a prompt to ask the LLM to refine its previous answer.
    """
    prompt = (
        "The following VQA pair was generated for a scene.\n\n"
        f"Question: \"{original_vqa['question']}\"\n"
        f"Initial Answer: \"{original_vqa['answer']}\"\n\n"
        "However, the initial answer has the following issue(s):\n"
    )
    for critique in critiques:
        prompt += f"- {critique}\n"
    prompt += "\nPlease rewrite the answer to be clear, correct, and free of these issues, based on the original question."
    return prompt


def main():
    """
    Main function to run the iterative VQA generation experiment.
    """
    print("===== Stage 1: Initial VQA Generation =====")

    # Mock scene data that would be derived from the VQASynth pipeline
    mock_scene_data = {
        "objects": [
            {"name": "red chair", "position": [1.5, 0, 2.0]},
            {"name": "blue sofa", "position": [-0.5, 0, 2.2]},
        ],
        "camera_viewpoint": "facing the furniture",
    }

    initial_prompt = (
        "You are a helpful assistant for a VQA data generation task. "
        "Based on the following scene summary, generate one spatial reasoning question and a detailed, descriptive answer.\n\n"
        f"Scene Summary: {mock_scene_data}\n\n"
        "Provide your output as a JSON object with 'question' and 'answer' keys. This is the initial question."
    )

    initial_vqa = mock_llm_call(initial_prompt)
    print("\n[Initial Generated VQA]")
    print(f"Question: {initial_vqa['question']}")
    print(f"Answer: {initial_vqa['answer']}\n")

    print("===== Stage 2: Validation =====")
    critiques = validate_vqa_pair(initial_vqa, mock_scene_data)

    if not critiques:
        print("Validation PASSED. The generated VQA pair is good.")
        final_vqa = initial_vqa
    else:
        print("Validation FAILED. Critiques found:")
        for c in critiques:
            print(f"- {c}")

        print("\n===== Stage 3: Iterative Refinement =====")
        refinement_prompt = build_refinement_prompt(initial_vqa, critiques)
        refined_vqa = mock_llm_call(refinement_prompt)

        print("\n[Refined VQA]")
        print(f"Question: {refined_vqa['question']}")
        print(f"Answer: {refined_vqa['answer']}\n")
        final_vqa = refined_vqa

    print("===== Experiment Complete =====")
    print("Final VQA Pair:", final_vqa)


if __name__ == "__main__":
    main()
