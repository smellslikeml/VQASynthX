import os
import json
import logging
from typing import List, Dict, Any


# Assume an LLM client is available, e.g., OpenAI
# This is a placeholder for the actual implementation in VQASynth
class LLMClient:
    def __init__(self, api_key):
        # In a real scenario, this would initialize an actual LLM client
        if not api_key:
            raise ValueError("API_KEY must be set for LLMClient.")
        self.api_key = api_key
        logging.info("LLMClient initialized.")

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Placeholder for actual LLM API call. This simulates the behavior for a test run.
        logging.info(
            f"Simulating LLM generation with prompt (first 100 chars): {prompt[:100]}..."
        )
        # Simulate a critique response if the prompt asks for it
        if "critique agent" in prompt.lower():
            return json.dumps(
                {
                    "critique": "The question 'Is object A near object B?' is too ambiguous. The answer format is inconsistent.",
                    "passed": False,
                    "suggestions": [
                        "Ask for specific metric distances.",
                        "Use a consistent JSON format for answers.",
                        "Increase question variety.",
                    ],
                }
            )
        # Simulate an initial, lower-quality VQA generation
        else:
            return json.dumps(
                {
                    "vqa_pairs": [
                        {
                            "question": "Is the red forklift near the brown boxes?",
                            "answer": "Yes, it is relatively close.",
                        },
                        {
                            "question": "What is to the left of the pallet?",
                            "answer": "A man in a red hat.",
                        },
                    ]
                }
            )


def build_generator_prompt(
    scene_data: Dict[str, Any], feedback: Dict[str, Any] = None
) -> str:
    """
    Constructs a prompt for the VQA generation LLM.
    If feedback is provided, it incorporates it to ask for corrections.
    This function mirrors the dynamic prompt logic in ChatBattery's `problem_conceptualization`.
    """
    objects_description = ", ".join(
        [
            f"{obj.get('name', 'unnamed object')}"
            for obj in scene_data.get("objects", [])
        ]
    )
    base_prompt = (
        f"You are a VQA data generation assistant. Based on the following scene, generate 5 high-quality "
        f"visual question-answering pairs focused on spatial relationships (e.g., distance, orientation, relative position).\n"
        f"Scene objects: {objects_description}\n"
        f"Output MUST be a single JSON object with a key 'vqa_pairs', which is a list of objects, "
        f"each with a 'question' and 'answer' key."
    )

    if feedback and not feedback.get("passed", True):
        # This mirrors the "update_with_generated_battery_list" mode in ChatBattery's main.py
        correction_prompt = (
            f"\n\nIn a previous attempt, your output had issues. Here is a critique: {feedback.get('critique', 'No critique available.')}\n"
            f"Suggestions for improvement: {'. '.join(feedback.get('suggestions', []))}\n"
            f"Please generate a new set of 5 VQA pairs that address this feedback. Do not repeat the previous mistakes."
        )
        return base_prompt + correction_prompt

    return base_prompt


def build_critique_prompt(generated_vqa_json: str) -> str:
    """
    Constructs a prompt for the critique LLM to evaluate the generated VQA pairs.
    This acts as the `Domain_Agent` or `Decision_Agent` from ChatBattery.
    """
    return (
        f"You are a VQA data quality critique agent. Evaluate the following generated VQA pairs. "
        f"Check for ambiguity, factual plausibility (assuming a generic scene), grammatical errors, and question diversity. "
        f"Your output MUST be a single JSON object with three keys:\n"
        f"1. 'critique' (a string summarizing the issues).\n"
        f"2. 'passed' (a boolean, true if the quality is high, false otherwise).\n"
        f"3. 'suggestions' (a list of strings for how to improve the generation).\n\n"
        f"VQA pairs to critique:\n{generated_vqa_json}"
    )


def process_image_data(
    image_id: str,
    scene_data: Dict[str, Any],
    llm_client: LLMClient,
    max_iterations: int = 2,
):
    """
    Main processing loop for a single image's scene data, now with an iterative refinement step.
    """
    logging.info(f"Processing image {image_id} with iterative VQA synthesis loop.")
    feedback = None
    final_vqa_pairs = []

    for i in range(max_iterations):
        logging.info(f"--- Iteration {i+1}/{max_iterations} for image {image_id} ---")

        # 1. Generate VQA pairs (inspired by ChatBattery's LLM_Agent)
        generator_prompt = build_generator_prompt(scene_data, feedback)
        generated_json_str = llm_client.generate(generator_prompt)

        # 2. Critique the generated pairs (inspired by ChatBattery's Domain_Agent/Search_Agent)
        critique_prompt = build_critique_prompt(generated_json_str)
        feedback_json_str = llm_client.generate(critique_prompt, temperature=0.2)

        try:
            feedback = json.loads(feedback_json_str)
            generated_data = json.loads(generated_json_str)
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to decode JSON from LLM for image {image_id}: {e}. Skipping."
            )
            return None

        logging.info(f"Critique for iteration {i+1}: {feedback}")

        if feedback.get("passed", False):
            logging.info(f"Critique passed. Finalizing VQA pairs for image {image_id}.")
            final_vqa_pairs = generated_data.get("vqa_pairs", [])
            break

        # If the loop finishes without passing, use the last generated pairs as a fallback.
        final_vqa_pairs = generated_data.get("vqa_pairs", [])

    return {
        "image_id": image_id,
        "final_vqa_pairs": final_vqa_pairs,
        "final_critique": feedback,
    }


if __name__ == "__main__":
    # In a real VQASynth pipeline, this would read/write from mounted volumes.
    # This main block serves as a test harness.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 1. Setup
    api_key = os.environ.get("OPENAI_API_KEY", "dummy-key")
    llm_client = LLMClient(api_key=api_key)

    # 2. Mock input data (would come from a previous stage's output file)
    mock_scene_data = {
        "image_warehouse_001": {
            "objects": [
                {"name": "red forklift"},
                {"name": "brown boxes"},
                {"name": "man in red hat"},
                {"name": "wooden pallet"},
            ]
        }
    }

    # 3. Process data
    output_data = []
    for image_id, scene_data in mock_scene_data.items():
        result = process_image_data(image_id, scene_data, llm_client, max_iterations=2)
        if result:
            output_data.append(result)

    # 4. Save output (would write to a file for the next stage)
    output_path = (
        "/app/data/output/refined_prompts.json"  # Example path in Docker container
    )
    logging.info(f"Writing final processed data to {output_path}")
    # In a real run, this file would be written. For this test, we print to stdout.
    print(json.dumps(output_data, indent=2))
