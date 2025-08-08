import json
import os
from pathlib import Path

# This is a conceptual placeholder for a call to an LLM API.
# In a real implementation, this would use a library like OpenAI, Anthropic, etc.
def generate_flawed_reasoning(correct_reasoning_json: str) -> str:
    """
    Uses a large language model to introduce a subtle flaw into a correct
    reasoning chain, inspired by the bug-injection prompt in APR-RL.
    """

    prompt = f"""You are an expert in spatial reasoning and logic, but you sometimes make subtle mistakes.
Your task is to take a correct reasoning chain about an image and **introduce one realistic logical or calculation flaw**. The flaw should be subtle and mimic a common human error.

1.  **Introduce a logical error**: Misinterpret a spatial relationship (e.g., confuse 'left of' with 'right of', 'in front of' with 'behind').
2.  **Introduce a calculation error**: Make a plausible mistake in unit conversion or distance estimation (e.g., mixing up feet and meters, a slight miscalculation).
3.  **Preserve the original structure and intent**: The reasoning should still look plausible at a glance.
4.  **Do not change the final answer drastically if possible**, the error should be in the 'how' not the 'what'.
5.  **Do not just add typos or grammatical errors**. The error must be in the logic.

Here is an example.
Input:
[Start of Correct Reasoning]
{{
    "question": "How far is the person in the red shirt from the blue car?",
    "correct_reasoning": "The person in the red shirt is approximately at a depth of 5 meters. The blue car is at a depth of 10 meters. The lateral distance between them seems to be about 3 meters. Using the Pythagorean theorem for the horizontal plane, the distance is sqrt((10-5)^2 + 3^2) = sqrt(25 + 9) = sqrt(34), which is approximately 5.8 meters.",
    "answer": "About 5.8 meters."
}}
[End of Correct Reasoning]

Output:
[Start of Flawed Reasoning]
{{
    "question": "How far is the person in the red shirt from the blue car?",
    "flawed_reasoning": "The person in the red shirt is approximately at a depth of 5 meters. The blue car is at a depth of 10 meters. The lateral distance between them seems to be about 3 meters. The total distance is the sum of the depth difference and lateral distance, so (10-5) + 3 = 8 meters.",
    "answer": "About 8 meters."
}}
[End of Flawed Reasoning]

Now, here is the correct reasoning chain you need to modify.

Input:
[Start of Correct Reasoning]
{correct_reasoning_json}
[End of Correct Reasoning]

Output:
[Start of Flawed Reasoning]
"""
    # In a real implementation, this prompt would be sent to an LLM API.
    # For this experiment, we will return a hardcoded flawed version.
    print("--- CONCEPTUAL LLM CALL ---")
    print(prompt)
    print("---------------------------")

    # Parsing the input JSON to create a hardcoded flawed response for demonstration.
    data = json.loads(correct_reasoning_json)
    data['flawed_reasoning'] = "The yellow chair is on the left side of the image. The wooden table is in the center. Based on the camera angle, the chair is positioned to the right of the table, not the left. Therefore, the statement is false."
    data.pop('correct_reasoning', None) # Remove the correct key

    return json.dumps(data)


def main():
    """Main function to generate a sample of flawed reasoning data."""
    output_dir = Path("experiments/reasoning_negatives/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "flawed_reasoning_samples.jsonl"

    # Sample input, similar to what VQASynth might produce.
    sample_input = {
        "image_id": "sample_001.jpg",
        "question": "Is the yellow chair to the left of the wooden table?",
        "correct_reasoning": "The yellow chair is on the left side of the image. The wooden table is in the center. From the perspective of the room, the chair is positioned to the left of the table. Therefore, the statement is true.",
        "answer": "Yes"
    }

    print(f"Processing sample: {sample_input['image_id']}")
    flawed_reasoning_str = generate_flawed_reasoning(json.dumps(sample_input))
    flawed_reasoning_obj = json.loads(flawed_reasoning_str)

    with open(output_file, 'w') as f:
        f.write(json.dumps(flawed_reasoning_obj) + '\n')

    print(f"\nSuccessfully generated flawed reasoning sample.")
    print(f"Output written to: {output_file}")
    print("\n--- Generated Content ---")
    print(json.dumps(flawed_reasoning_obj, indent=2))


if __name__ == "__main__":
    main()
