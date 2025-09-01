import os
import json
import openai
from textwrap import dedent

# --- Configuration ---
# Ensure the OPENAI_API_KEY environment variable is set before running.
# export OPENAI_API_KEY='your_key_here'

# --- Mock Data (Simulating output from VQASynth's scene fusion stage) ---
MOCK_SCENE_DESCRIPTION = {
    "image_id": "warehouse_sample_1.jpeg",
    "objects": [
        {
            "id": 1,
            "label": "red forklift",
            "position_2d": [150, 400],
            "estimated_distance_m": 5.5,
        },
        {
            "id": 2,
            "label": "brown cardboard boxes",
            "position_2d": [600, 450],
            "estimated_distance_m": 7.0,
        },
        {
            "id": 3,
            "label": "wooden pallet",
            "position_2d": [550, 600],
            "estimated_distance_m": 7.2,
        },
        {
            "id": 4,
            "label": "man in red hat",
            "position_2d": [300, 500],
            "estimated_distance_m": 8.1,
        },
    ],
    "relationships": [
        {"object1_id": 1, "relation": "left of", "object2_id": 2},
        {"object1_id": 4, "relation": "near", "object2_id": 3},
    ],
    "summary": "A warehouse scene containing a red forklift, stacked cardboard boxes, a wooden pallet, and a man in a red hat.",
}

TOPICS = [
    "Warehouse Logistics & Safety",
    "Inventory Management",
    "Robotics Path Planning",
]

# --- Prompt Template (Inspired by ProactiveEval's data_generation_prompt.py) ---
SCENARIO_GENERATION_PROMPT_TEMPLATE = dedent(
    """
    You are an expert in generating realistic scenarios for visual question answering tasks.
    Your goal is to create a plausible user scenario and a specific target based on a scene description and a high-level topic. The scenario should set a context, and the target should define a clear, actionable goal for a user interacting with an AI assistant.

    Respond with a single JSON object with two keys: "scenario" and "target". Do not add any other text or explanations.

    **Topic:**
    {topic}

    **Scene Description:**
    ```json
    {scene_description}
    ```

    **JSON Output:**
"""
)


def generate_scenario_and_target(client, topic, scene_description):
    """Generates a scenario and target using an LLM, inspired by ProactiveEval's generation process."""
    prompt = SCENARIO_GENERATION_PROMPT_TEMPLATE.format(
        topic=topic, scene_description=json.dumps(scene_description, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that only outputs JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def format_vqa_pair(scene_data, scenario_data):
    """Formats the generated scenario into a conversational VQA pair."""
    question = f'USER: I am working on a task related to {scenario_data.get("scenario", "the scene")}. My goal is to {scenario_data.get("target", "understand the scene better")}. Based on the visual information, can you help me?'
    answer = f'ASSISTANT: (Thought process: The user needs help with their goal: \'{scenario_data.get("target")}\'. I need to analyze the objects {[(obj["label"]) for obj in scene_data["objects"]]} and their relationships to provide a helpful response...)'
    return question, answer


def main():
    """Main function to run the experiment."""
    print("Initializing Scenario-Driven VQA Synthesis Experiment...")

    try:
        client = openai.OpenAI()
        # Test API connection
        client.models.list()
    except openai.AuthenticationError:
        print("\n--- ERROR ---")
        print("OpenAI API key not found or invalid.")
        print("Please set the OPENAI_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred during client initialization: {e}")
        return

    print(
        f"Successfully initialized OpenAI client. Processing {len(TOPICS)} topics...\n"
    )

    for topic in TOPICS:
        print(f"--- Generating for Topic: {topic} ---")
        result = generate_scenario_and_target(client, topic, MOCK_SCENE_DESCRIPTION)

        if result and "scenario" in result and "target" in result:
            print(f"  Generated Scenario: {result['scenario']}")
            print(f"  Generated Target: {result['target']}\n")

            question, answer_cot = format_vqa_pair(MOCK_SCENE_DESCRIPTION, result)
            print("  Example VQA Pair:")
            print(f"    {question}")
            print(f"    {answer_cot}\n")
        else:
            print("  Failed to generate a valid scenario for this topic.\n")


if __name__ == "__main__":
    main()
