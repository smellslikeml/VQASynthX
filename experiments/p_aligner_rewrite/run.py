import os
import json
from openai import OpenAI

# This is the core idea from P-Aligner: a meta-prompt to refine an existing prompt.
# Evidence: https://github.com/F2-Song/P-Aligner/blob/main/prepare_data.py
PROMPT_ALIGNER_TEMPLATE = """You are an expert prompt engineer. Please help me optimize this prompt to get better response:

[The Start of Raw Prompt]
{raw_prompt}
[The End of Raw Prompt]"""

# The expected response format, which we need to parse.
# Evidence: https://github.com/F2-Song/P-Aligner/blob/main/prepare_data.py
RESPONSE_PREFIX = "The Optimized Prompt:\n\n[The Start of Optimized Prompt]\n"
RESPONSE_SUFFIX = "\n[The End of Optimized Prompt]"

# Sample VQA questions typical of the VQASynth project.
# These will serve as our "raw prompts" to be optimized.
SAMPLE_VQA_QUESTIONS = [
    "How close is the man in red hat walking from the wooden pallet with boxes?",
    "Does the red forklift in warehouse appear on the left side of the brown cardboard boxes stacked?",
    "Is the person in the blue shirt standing closer to the camera than the shelves in the background?",
    "Estimate the distance in feet between the two white vans.",
    "What is the spatial relationship between the laptop and the coffee mug on the desk?",
]

def align_prompt(raw_prompt: str, client: OpenAI) -> str:
    """
    Uses an LLM to rewrite a raw prompt based on the P-Aligner methodology.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo",  # A powerful model is suitable for prompt engineering
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_ALIGNER_TEMPLATE.format(raw_prompt=raw_prompt),
                }
            ],
            temperature=0.2, # Lower temperature for more deterministic and focused rewriting
        )
        response_content = completion.choices[0].message.content

        # Parse the optimized prompt from the model's response
        if RESPONSE_PREFIX in response_content:
            aligned_prompt = response_content.split(RESPONSE_PREFIX)[1]
            if RESPONSE_SUFFIX in aligned_prompt:
                aligned_prompt = aligned_prompt.split(RESPONSE_SUFFIX)[0]
            return aligned_prompt.strip()
        else:
            # If the model doesn't follow the format, return its full response for debugging
            return f"Failed to parse response: {response_content}"

    except Exception as e:
        return f"An error occurred: {e}"

def main():
    """
    Main function to run the P-Aligner experiment.
    """
    print("--- Running VQA Prompt Alignment Experiment (inspired by P-Aligner) ---")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this experiment.")
        return

    client = OpenAI(api_key=api_key)

    results = []
    for i, question in enumerate(SAMPLE_VQA_QUESTIONS):
        print(f"\n--- Optimizing Question {i+1}/{len(SAMPLE_VQA_QUESTIONS)} ---")
        print(f"Original: {question}")
        
        aligned_question = align_prompt(question, client)
        
        print(f"Aligned:  {aligned_question}")
        results.append({
            "original_question": question,
            "aligned_question": aligned_question,
        })
    
    output_path = "aligned_vqa_prompts.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n--- Experiment Complete ---")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
