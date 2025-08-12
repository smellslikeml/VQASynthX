import os
import json


# This function is a mock-up of an LLM API call.
def get_llm_response(prompt):
    """
    A placeholder for an LLM API call.
    In a real implementation, this would connect to an LLM service.
    For this self-contained example, it will return pre-defined responses
    to demonstrate the flow without requiring an API key.
    """
    print("=" * 20 + " PROMPT TO LLM " + "=" * 20)
    print(prompt)
    print("=" * 55)

    # Mock responses to simulate the LLM's behavior
    if "generate five distinct spatial reasoning questions" in prompt:
        response = """
Here are five questions based on the scene:
* Is the chair to the left of the table?
* What is on the table?
* How far is the plant from the window?
* Is the book on the floor?
* Is the ceiling lamp directly above the table?
"""
    elif "Critique the following questions" in prompt:
        response = """
Here is a critique of the provided questions:
* "Is the chair to the left of the table?": Good, simple relative position question.
* "What is on the table?": Weak. This is an object identification question, not a spatial reasoning one.
* "How far is the plant from the window?": Good, but could be more specific (e.g., metric distance).
* "Is the book on the floor?": Ambiguous. 'On the floor' is not specific. A better question would relate the book to another object.
* "Is the ceiling lamp directly above the table?": Good, checks for vertical alignment.
"""
    elif "Refine the original questions based on the provided critique" in prompt:
        response = """
Here are the refined questions, addressing the critique:
* Is the arm of the chair closer to the window than the table leg is?
* What is the spatial relationship between the laptop and the coffee mug on the table?
* Estimate the distance in feet between the base of the plant pot and the window sill.
* Is the red book on the floor located between the chair and the table?
* From the camera's perspective, does the ceiling lamp occlude any part of the window?
"""
    else:
        response = "Unknown prompt type. No mock response available."

    print("=" * 20 + " MOCK LLM RESPONSE " + "=" * 18)
    print(response)
    print("=" * 55 + "\n")
    return response


def parse_questions(llm_output):
    """Extracts bulleted questions from LLM text output."""
    return [
        line.strip("* ").strip()
        for line in llm_output.strip().split("\n")
        if line.strip().startswith("*")
    ]


def problem_conceptualization_initial(scene_description_text):
    """
    Creates the initial prompt for question generation, inspired by ChatBattery's
    'initial' mode prompt template.
    """
    return f"""
Based on the following scene description, generate five distinct spatial reasoning questions.
The questions should test concepts like relative positioning, distance estimation, and object orientation.
List them in bullet points (using an asterisk *).

Scene Description:
{scene_description_text}
"""


def problem_conceptualization_refinement(original_questions, critique):
    """
    Creates the refinement prompt, inspired by ChatBattery's 'update_with_generated_battery_list'
    which incorporates feedback to generate better results.
    """
    return f"""
You previously generated a list of spatial questions. A critique has identified some as weak, ambiguous, or not focused on spatial reasoning.
Refine the original questions based on the provided critique to make them more challenging and precise.
The new questions must address the specific feedback points.
List the five new, improved questions in bullet points (using an asterisk *).

Original Questions:
{original_questions}

Critique:
{critique}
"""


def run_experiment():
    """
    Main function to run the iterative VQA generation experiment.
    """
    # Step 0: Define a scene description. In the full VQASynth pipeline,
    # this would be generated from an image.
    scene_description = {
        "objects": [
            {"name": "table", "position": [0, 0, 0]},
            {"name": "chair", "position": [-1.5, 0, 0]},
            {"name": "plant", "position": [2.5, 0, 1.2]},
            {"name": "window", "position": [3, 0, 1.5]},
            {"name": "book", "position": [-0.5, -1.0, 0]},
            {"name": "ceiling lamp", "position": [0, 0, 2.5]},
        ],
        "relations": [
            "The chair is to the left of the table.",
            "The table has a laptop and a coffee mug on it.",
            "The plant is near the window.",
        ],
    }
    scene_description_text = json.dumps(scene_description, indent=2)

    print("### STAGE 1: INITIAL QUESTION GENERATION ###\n")
    initial_prompt = problem_conceptualization_initial(scene_description_text)
    initial_response = get_llm_response(initial_prompt)
    initial_questions = parse_questions(initial_response)

    print("### STAGE 2: CRITIQUE GENERATION ###\n")
    critique_prompt = f"""
Critique the following questions based on their quality for testing spatial reasoning.
Identify weaknesses such as ambiguity, lack of spatial focus, or simplicity.

Questions to Critique:
{initial_response}
"""
    critique_response = get_llm_response(critique_prompt)

    print("### STAGE 3: REFINEMENT GENERATION ###\n")
    refinement_prompt = problem_conceptualization_refinement(
        initial_response, critique_response
    )
    refinement_response = get_llm_response(refinement_prompt)
    refined_questions = parse_questions(refinement_response)

    print("### EXPERIMENT COMPLETE ###")
    print("\n--- Initial Questions ---")
    for q in initial_questions:
        print(f"- {q}")
    print("\n--- Refined Questions ---")
    for q in refined_questions:
        print(f"- {q}")


if __name__ == "__main__":
    run_experiment()
