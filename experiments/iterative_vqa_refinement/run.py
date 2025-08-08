import os
import json
import openai

# --- Configuration ---
# In a real environment, this would be managed more securely.
# For this experiment, we rely on an environment variable:
# `export OPENAI_API_KEY='your_key_here'`
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# --- Mock Scene Data ---
# In a real integration, this would come from vqasynth's scene analysis pipeline.
MOCK_SCENE_CONTEXT = {
    "image_id": "warehouse_sample_1.jpeg",
    "description": "A warehouse interior.",
    "objects": [
        {"id": 1, "class": "red forklift", "location": "center-left"},
        {"id": 2, "class": "stack of brown cardboard boxes", "location": "center-right"},
        {"id": 3, "class": "man in red hat", "location": "background-left"},
        {"id": 4, "class": "wooden pallet with boxes", "location": "foreground-right"},
    ]
}

# --- Agent Prompts (Inspired by ChatBattery's iterative process) ---

def get_initial_generation_prompt(scene_context):
    """
    Creates the prompt for the initial VQA question generation.
    Analogous to ChatBattery's initial material proposal.
    """
    object_list = "\n".join([f"- {obj['class']} at {obj['location']}" for obj in scene_context['objects']])
    return f"""
    Given the following scene context from an image of a warehouse:
    {object_list}

    Your task is to generate 5 diverse spatial reasoning questions about the relationships between these objects.
    - Questions should be specific.
    - Questions should require reasoning about relative positions (left of, behind, etc.).

    List the questions in a JSON array of strings. Do not add any other text or markdown fences.
    Example format: ["Is the red forklift to the left of the pallet?", "How many boxes are stacked?"]
    """

def get_validation_prompt(questions, scene_context):
    """
    Creates the prompt for the Validator Agent.
    This agent critiques the generated questions.
    """
    question_list = "\n".join([f"- {q}" for q in questions])
    object_list = "\n".join([f"- {obj['class']}" for obj in scene_context['objects']])
    return f"""
    You are a critique agent. Your task is to validate a list of generated questions based on a scene context.
    
    Scene Objects:
    {object_list}

    Generated Questions:
    {question_list}

    Analyze each question and identify flaws. Categorize flaws as 'AMBIGUOUS' (e.g., refers to "the box" when there are multiple), 'TRIVIAL' (e.g., a simple yes/no about presence), or 'GOOD'.
    
    Return your critique as a JSON object where keys are the questions and values are an object with a "critique" and "category".

    Example format:
    {{
        "Is the red forklift to the left of the pallet?": {{ "critique": "Good, specific question about relative position.", "category": "GOOD" }},
        "Is there a forklift?": {{ "critique": "Trivial question, the context already lists a forklift.", "category": "TRIVIAL" }}
    }}
    """

def get_refinement_prompt(critiqued_questions, scene_context):
    """
    Creates the prompt for the Refiner Agent.
    Analogous to ChatBattery's `update_with_generated_battery_list`.
    """
    object_list = "\n".join([f"- {obj['class']}" for obj in scene_context['objects']])
    
    bad_questions_critique = ""
    for q, data in critiqued_questions.items():
        if data['category'] != 'GOOD':
            bad_questions_critique += f"- Question: \"{q}\"\n  - Flaw ({data['category']}): {data['critique']}\n"
    
    if not bad_questions_critique:
        return None # All questions are good

    return f"""
    You previously generated some spatial reasoning questions. Some of them had flaws.
    Your task is to generate new, improved questions to replace ONLY the flawed ones.

    Scene Context:
    {object_list}

    Flawed Questions and Critiques:
    {bad_questions_critique}

    Instructions for new questions:
    - Avoid the identified flaws.
    - Make questions more specific and require more detailed spatial reasoning.
    - For ambiguous questions, refer to objects more precisely.
    - For trivial questions, ask about relationships, not just existence.

    Return a JSON array of strings containing ONLY the new, replacement questions. The number of new questions should match the number of flawed questions. Do not add any other text or markdown fences.
    """

# --- LLM Interaction Wrapper ---
def ask_llm(prompt, model="gpt-4-turbo"):
    """A simple wrapper for the OpenAI API call."""
    raw_content = ""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={ "type": "json_object" },
        )
        raw_content = response.choices[0].message.content
        # The API may wrap the JSON in a markdown fence despite the format request
        if raw_content.startswith("```json"):
            raw_content = raw_content.strip("```json\n").rstrip("\n```")
        return json.loads(raw_content)
    except Exception as e:
        print(f"Error calling LLM or parsing JSON: {e}")
        print(f"LLM raw response was:\n{raw_content}")
        return None


# --- Main Experiment Logic ---
def run_experiment():
    """
    Orchestrates the iterative refinement process.
    """
    print("--- 1. Initial Question Generation ---")
    initial_prompt = get_initial_generation_prompt(MOCK_SCENE_CONTEXT)
    response_data = ask_llm(initial_prompt)
    initial_questions = response_data if isinstance(response_data, list) else list(response_data.values())[0]

    if not initial_questions:
        print("Failed to generate initial questions. Exiting.")
        return
    
    with open("initial_questions.json", "w") as f:
        json.dump(initial_questions, f, indent=2)
    print(f"Generated {len(initial_questions)} initial questions.")
    print(json.dumps(initial_questions, indent=2))


    print("\n--- 2. Validation Step ---")
    validation_prompt = get_validation_prompt(initial_questions, MOCK_SCENE_CONTEXT)
    critique = ask_llm(validation_prompt)
    if not critique:
        print("Failed to generate critique. Exiting.")
        return

    with open("validated_questions.json", "w") as f:
        json.dump(critique, f, indent=2)
    print("Generated critique for questions.")
    print(json.dumps(critique, indent=2))

    good_questions = [q for q, data in critique.items() if data.get('category') == 'GOOD']
    
    print("\n--- 3. Refinement Step ---")
    refinement_prompt = get_refinement_prompt(critique, MOCK_SCENE_CONTEXT)
    
    if refinement_prompt is None:
        print("All questions were validated as GOOD. No refinement needed.")
        final_questions = initial_questions
    else:
        response_data = ask_llm(refinement_prompt)
        refined_questions = response_data if isinstance(response_data, list) else list(response_data.values())[0]

        if not refined_questions:
            print("Failed to generate refined questions. Using only the good ones from the initial batch.")
            final_questions = good_questions
        else:
            print(f"Generated {len(refined_questions)} new questions to replace flawed ones.")
            print(json.dumps(refined_questions, indent=2))
            final_questions = good_questions + refined_questions

    with open("refined_questions.json", "w") as f:
        json.dump(final_questions, f, indent=2)
    
    print("\n--- Experiment Complete ---")
    print("Final set of questions saved to refined_questions.json:")
    print(json.dumps(final_questions, indent=2))

if __name__ == "__main__":
    run_experiment()
