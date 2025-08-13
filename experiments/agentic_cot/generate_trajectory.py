import os
import json
from typing import List, Dict, Any


# A mock LLM client for demonstration purposes.
# In a real implementation, this would use a library like openai, anthropic, etc.
class MockLLMClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required.")
        self.api_key = api_key
        self.call_count = 0

    def generate(self, prompt: str, max_steps: int) -> str:
        """Simulates an LLM call. Returns a structured response."""
        # This mock will produce a fixed, multi-step reasoning process.
        responses = [
            '[THINK] I need to identify the objects mentioned in the query: "red forklift" and "brown cardboard boxes". I will first locate the red forklift in the scene description.',
            '[THINK] The forklift is located at coordinates (10, 5, 0). Now I need to locate the brown cardboard boxes. The scene description mentions "a stack of brown boxes" at coordinates (15, 5, 0).',
            "[THINK] The query asks if the forklift is to the left of the boxes. I need to compare their x-coordinates. The forklift is at x=10 and the boxes are at x=15. Since 10 < 15, the forklift is indeed to the left of the boxes.",
            "[ANSWER] Yes, the red forklift appears on the left side of the brown cardboard boxes.",
        ]
        if self.call_count < len(responses):
            response = responses[self.call_count]
            self.call_count += 1
            return response
        return "[ANSWER] I have completed the reasoning process."


# Initialize the mock client
# In a real setup, this would come from environment variables or a config file.
LLM_CLIENT = MockLLMClient(api_key=os.environ.get("OPENAI_API_KEY", "mock_key"))

SYSTEM_PROMPT = """
You are an expert spatial reasoning agent. Your goal is to answer a question about a scene.
To do this, you must think step-by-step. Use the [THINK] tool to outline your reasoning process.
Each thought should be a single, logical step.
When you are confident in your final answer, use the [ANSWER] tool to provide it.
Do not provide the answer until you have finished thinking.

Example:
Scene: A cat is on a mat. The cat is at (1,1). The mat is at (1,0).
Question: Is the cat on the mat?

[THINK] The cat is at (1,1) and the mat is at (1,0). The query is about their spatial relationship.
[THINK] "On" implies that the cat's Z coordinate should be higher than the mat's, and their X,Y coordinates should be similar. Let's assume Z is the vertical axis. The problem description does not give Z coordinates, but the description "on a mat" is the primary source of truth. Let's check the provided coordinates.
[THINK] The coordinates are (1,1) and (1,0). These are 2D coordinates. Assuming the context "on a mat" is the primary source of truth, I will proceed with that.
[ANSWER] Yes, the cat is on the mat.
"""


def generate_reasoning_trajectory(
    scene_description: str, question: str, max_steps: int = 5
) -> Dict[str, Any]:
    """
    Generates a structured reasoning trajectory for a given scene and question.
    This mimics the sequential tool use and trajectory recording from Trae-Agent.

    Args:
        scene_description: Textual description of the scene context.
        question: The user's question.
        max_steps: The maximum number of reasoning steps to take.

    Returns:
        A dictionary containing the reasoning trajectory and the final answer.
    """
    trajectory = []
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\nScene: {scene_description}\nQuestion: {question}\n"
    )

    for i in range(max_steps):
        # In a real scenario, you would pass the growing context to the LLM.
        # For simplicity in this mock, the client simulates the steps.
        llm_response = LLM_CLIENT.generate(full_prompt, max_steps)

        if llm_response.strip().startswith("[THINK]"):
            thought = llm_response.strip().replace("[THINK]", "").strip()
            trajectory.append(thought)
            # Append the thought to the prompt for the next turn
            full_prompt += f"\n{llm_response}"
        elif llm_response.strip().startswith("[ANSWER]"):
            answer = llm_response.strip().replace("[ANSWER]", "").strip()
            return {
                "question": question,
                "reasoning_trajectory": trajectory,
                "answer": answer,
            }
        else:
            # Handle cases where the LLM doesn't follow the format
            return {
                "question": question,
                "reasoning_trajectory": trajectory,
                "answer": llm_response,
                "error": "LLM did not follow the expected [THINK]/[ANSWER] format.",
            }

    return {
        "question": question,
        "reasoning_trajectory": trajectory,
        "answer": "Failed to reach an answer within the maximum number of steps.",
        "error": f"Max steps ({max_steps}) reached.",
    }


if __name__ == "__main__":
    # Example usage based on VQASynth's domain
    mock_scene_description = (
        "A warehouse scene. A red forklift is at 3D coordinates (10, 5, 0). "
        "A stack of brown cardboard boxes is at 3D coordinates (15, 5, 0). "
        "A man in a red hat is at (2, 8, 0)."
    )

    mock_question = "Does the red forklift in the warehouse appear on the left side of the brown cardboard boxes?"

    # The mock client needs to be reset for each run in this test script
    LLM_CLIENT.call_count = 0
    result = generate_reasoning_trajectory(mock_scene_description, mock_question)

    print(json.dumps(result, indent=2))
