import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm

# Inspired by ChatBattery's use of OpenAI for hypothesis generation
# and its iterative refinement loop based on feedback.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_refinement_prompt(vqa_pair_json_str: str) -> str:
    """Creates the prompt for the LLM to act as a VQA Quality Expert."""
    # This prompt structure is inspired by the problem_conceptualization function
    # in ChatBattery's main.py, which sets a clear role, provides rules,
    # and gives specific instructions for the desired output format.
    return f"""You are a VQA Quality Expert. Your task is to refine a Visual Question Answering pair to create high-quality training data for a spatial reasoning Vision Language Model.

A good spatial reasoning question is:
- Unambiguous: It refers to specific objects, especially if multiple similar objects exist (e.g., 'the red chair on the left' instead of 'the chair').
- Spatially Focused: It probes relationships like distance, orientation, or relative position.
- Non-Trivial: It requires genuine reasoning, not just stating the obvious (e.g., avoid 'Is the grass green?').
- Answerable: The answer must be derivable from the provided context.

Critique the following VQA pair. If it is already high-quality, you can return it as-is. If it has flaws, please rewrite the 'question' to be better, while keeping the answer consistent with the provided context.

Your response MUST be a single JSON object with one key, "refined_vqa", containing the refined VQA json.

Original VQA Pair:
{vqa_pair_json_str}

Respond with the JSON object:
"""


def refine_vqa_pair(vqa_pair: dict) -> dict:
    """Sends a VQA pair to the LLM for refinement and returns the improved version."""
    try:
        vqa_pair_str = json.dumps(vqa_pair, indent=2)
        prompt = get_refinement_prompt(vqa_pair_str)

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that only responds with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        content = response.choices[0].message.content
        refined_data = json.loads(content)
        return refined_data.get(
            "refined_vqa", vqa_pair
        )  # Return original if key is missing

    except Exception as e:
        print(
            f"An error occurred while refining VQA pair: {vqa_pair.get('id', 'N/A')}. Error: {e}"
        )
        return vqa_pair  # Return the original pair in case of an API or parsing error


def main():
    parser = argparse.ArgumentParser(
        description="Refine VQA pairs using an LLM critic."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input JSONL file with VQA pairs.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output JSONL file for refined VQA pairs.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    refined_pairs = []
    with open(args.input_file, "r") as f_in:
        vqa_pairs = [json.loads(line) for line in f_in]

        for vqa_pair in tqdm(vqa_pairs, desc="Refining VQA Pairs"):
            refined_pair = refine_vqa_pair(vqa_pair)
            refined_pairs.append(refined_pair)

    with open(args.output_file, "w") as f_out:
        for pair in refined_pairs:
            f_out.write(json.dumps(pair) + "\n")

    print(
        f"Refinement complete. Saved {len(refined_pairs)} pairs to {args.output_file}"
    )


if __name__ == "__main__":
    main()
