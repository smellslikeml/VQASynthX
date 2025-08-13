import os
import json
from typing import Dict, Any


# --- LLM Simulation ---
# In a real implementation, this would be a call to an actual LLM
# (e.g., via Ollama, OpenAI API, etc.). For this self-contained example,
# we simulate the LLM's response based on simple logic.
def call_llm(prompt: str, context: Dict[str, Any]) -> str:
    """Simulates a call to an LLM for VQA generation and validation."""
    print(f"\n--- LLM Input ---\n{prompt}\n-----------------\n")

    # Simulate Generator Agent
    if "GENERATE VQA" in prompt:
        # Initial, slightly flawed generation
        if context.get("revision_count", 0) == 0:
            response = {
                "question": "Is the red block to the left of the blue sphere?",
                "reasoning": "The red block is at x=2 and the blue sphere is at x=5. The left direction is decreasing x. Therefore, the red block is to the left of the blue sphere.",
                "answer": "Yes.",
                "critique_request": "Critique this VQA pair for factual accuracy and clarity.",
            }
        # Revised, improved generation
        else:
            response = {
                "question": "Is the red block located to the left of the blue sphere on the table?",
                "reasoning": "The scene contains a red block at coordinate x=2 and a blue sphere at coordinate x=5. In this coordinate system, smaller x values are to the left. Since 2 < 5, the red block is indeed to the left of the blue sphere. Both objects are on the table.",
                "answer": "Yes",
            }
        return json.dumps(response, indent=2)

    # Simulate Validator Agent
    elif "VALIDATE VQA" in prompt:
        generated_vqa = context["generated_vqa"]
        # Initial critique
        if generated_vqa["answer"] == "Yes.":  # Simple check for the flawed version
            response = {
                "is_valid": False,
                "feedback": "The reasoning is correct but lacks detail. The question could be more specific by mentioning the shared surface ('the table'). The answer 'Yes.' is too blunt; a single word 'Yes' is preferred for VQA datasets.",
                "revised_vqa": None,
            }
        # Approval of the revised version
        else:
            response = {
                "is_valid": True,
                "feedback": "The revised VQA pair is factually accurate, the reasoning is detailed, and the question is specific. The answer format is correct. Approved.",
                "revised_vqa": generated_vqa,
            }
        return json.dumps(response, indent=2)

    return "{'error': 'Unknown prompt type'}"


# --- Prompt Templates (Inspired by empathia/code/assessment_prompts.py) ---

GENERATOR_PROMPT_TEMPLATE = """
**SYSTEM**: You are a VQA Generator Agent. Your task is to create a high-quality spatial reasoning question-answer pair based on the provided scene data.
Your output must be a JSON object with keys: "question", "reasoning", "answer".

**CONTEXT**:
{scene_data}

**TASK**: GENERATE VQA.
Based on the scene data, create a question about the relative positions of two objects. Provide a step-by-step reasoning process and a concise final answer.
"""

REVISION_PROMPT_TEMPLATE = """
**SYSTEM**: You are a VQA Generator Agent. Your previous generation was critiqued by a validator. You must revise your output based on the feedback.
Your output must be a JSON object with keys: "question", "reasoning", "answer".

**CONTEXT**:
{scene_data}

**PREVIOUS VQA**:
{previous_vqa}

**VALIDATOR FEEDBACK**:
{feedback}

**TASK**: REVISE VQA.
Regenerate the VQA pair, directly addressing all points in the validator's feedback.
"""

VALIDATOR_PROMPT_TEMPLATE = """
**SYSTEM**: You are a VQA Validator Agent. Your task is to critique a VQA pair for quality, factual accuracy, and adherence to formatting standards.
Your output must be a JSON object with keys: "is_valid" (boolean), "feedback" (string), and "revised_vqa" (the final JSON object if valid, otherwise null).

**CRITERIA**:
1.  **Factual Accuracy**: Is the answer and reasoning perfectly consistent with the scene data?
2.  **Clarity**: Is the question unambiguous? Is the reasoning easy to follow?
3.  **Format**: Is the answer a single word (e.g., "Yes", "No") or a short phrase as required?

**CONTEXT**:
{scene_data}

**VQA TO VALIDATE**:
{generated_vqa}

**TASK**: VALIDATE VQA.
Assess the provided VQA pair against the criteria and provide structured feedback.
"""


def run_synthesis_cycle(scene_data: Dict[str, Any], max_revisions: int = 2):
    """
    Implements the generator-validator loop inspired by EMPATHIA's architecture.
    """
    print("🚀 Starting VQA Synthesis Cycle...")
    print(f"Scene Data: {json.dumps(scene_data, indent=2)}")

    revision_count = 0
    generated_vqa = None
    is_valid = False

    while not is_valid and revision_count <= max_revisions:
        # 1. GENERATION / REVISION
        if revision_count == 0:
            print(f"\n--- Iteration {revision_count + 1}: Initial Generation ---")
            prompt = GENERATOR_PROMPT_TEMPLATE.format(scene_data=json.dumps(scene_data))
            context = {"revision_count": revision_count}
        else:
            print(f"\n--- Iteration {revision_count + 1}: Revision ---")
            prompt = REVISION_PROMPT_TEMPLATE.format(
                scene_data=json.dumps(scene_data),
                previous_vqa=json.dumps(generated_vqa, indent=2),
                feedback=validation_result["feedback"],
            )
            context = {"revision_count": revision_count}

        llm_response_str = call_llm(prompt, context)
        generated_vqa = json.loads(llm_response_str)
        print(f"Generator Agent Output:\n{json.dumps(generated_vqa, indent=2)}")

        # 2. VALIDATION
        print("\n--- Sending to Validator Agent ---")
        prompt = VALIDATOR_PROMPT_TEMPLATE.format(
            scene_data=json.dumps(scene_data),
            generated_vqa=json.dumps(generated_vqa, indent=2),
        )
        context = {"generated_vqa": generated_vqa}
        llm_response_str = call_llm(prompt, context)
        validation_result = json.loads(llm_response_str)
        print(f"Validator Agent Output:\n{json.dumps(validation_result, indent=2)}")

        is_valid = validation_result["is_valid"]
        if not is_valid:
            revision_count += 1
        else:
            final_vqa = validation_result["revised_vqa"]
            print("\n✅ Validation Successful!")
            print(f"Final VQA Pair:\n{json.dumps(final_vqa, indent=2)}")
            return final_vqa

    if not is_valid:
        print(f"\n❌ Validation Failed after {max_revisions} revisions.")
        return None


if __name__ == "__main__":
    # Define the input scene data for the VQA generation pipeline.
    # This would typically come from an upstream stage in VQASynth.
    scene_data = {
        "image_id": "warehouse_sample_1.jpeg",
        "objects": [
            {"id": "obj1", "class": "red block", "position": [2, 3, 1]},
            {"id": "obj2", "class": "blue sphere", "position": [5, 3, 1]},
            {"id": "obj3", "class": "table", "surface": True},
        ],
        "relationships": ["obj1 is on obj3", "obj2 is on obj3"],
    }
    run_synthesis_cycle(scene_data)
