import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_FILE = "synthetic_spatial_errors.jsonl"
SPATIAL_PROBLEMS = [
    {
        "id": "problem_1",
        "text": "In a photo, a red ball is 5 feet to the left of a blue cube. The blue cube is 3 feet to the left of a green cone. How far is the red ball from the green cone?",
    },
    {
        "id": "problem_2",
        "text": "You are facing North. You walk 10 meters forward, turn right and walk 10 meters, then turn right again and walk 5 meters. In which direction are you facing, and what direction is your starting point from your current location?",
    },
    {
        "id": "problem_3",
        "text": "A map shows that Town A is 20 miles directly west of Town C. Town B is 15 miles directly south of Town A. Is Town B to the northwest or southwest of Town C?",
    },
    {
        "id": "problem_4",
        "text": "On a shelf, a book is to the right of a vase. A candle is to the left of the vase. A picture frame is to the left of the candle. Which object is in the middle?",
    },
]


def get_prompt_template():
    """Creates the instruction prompt for the language model."""
    return (
        "You are an expert in cognitive science and spatial reasoning. Your task is to simulate human problem-solving, including common errors."
        " For the following spatial problem, provide two distinct responses in a single JSON object:\n"
        "1. **Correct Answer**: The accurate solution with a clear, step-by-step explanation.\n"
        "2. **Plausible Mistake**: An incorrect answer that a human might plausibly make due to a common reasoning error (e.g., mixing up left/right, adding distances incorrectly, misinterpreting a relative position). Explain the flawed reasoning process that leads to this mistake.\n\n"
        "Problem: {problem_text}\n\n"
        "Respond ONLY with a valid JSON object formatted like this:\n"
        '{"correct_answer": {"solution": "...", "explanation": "..."}, "plausible_mistake": {"solution": "...", "explanation": "..."}}'
    )


def main():
    """Main function to run the experiment."""
    print("Starting plausible error synthesis experiment...")

    # Authenticate with Hugging Face Hub
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError(
            "HUGGING_FACE_HUB_TOKEN environment variable not set. Please provide your token."
        )
    login(token=hf_token)

    # Load model and tokenizer, similar to the pattern in the source repo's get_models.py
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Create a text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    prompt_template = get_prompt_template()

    # Process each problem and write to a JSONL file
    with open(OUTPUT_FILE, "w") as f:
        for i, problem in enumerate(SPATIAL_PROBLEMS):
            print(f"Processing problem {i+1}/{len(SPATIAL_PROBLEMS)}: {problem['id']}")

            # Using the chat template for Llama 3 Instruct
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that only responds in valid JSON format.",
                },
                {
                    "role": "user",
                    "content": prompt_template.format(problem_text=problem["text"]),
                },
            ]

            # Generate the response
            outputs = pipe(messages, pad_token_id=tokenizer.eos_token_id)
            generated_text = outputs[0]["generated_text"][-1]["content"]

            # Attempt to parse the JSON response
            try:
                # Clean up potential markdown fences
                if generated_text.strip().startswith("```json"):
                    generated_text = generated_text.strip()[7:-4]

                response_json = json.loads(generated_text)

                # Create the record to save
                record = {
                    "problem_id": problem["id"],
                    "problem_text": problem["text"],
                    "model_output": response_json,
                }
                f.write(json.dumps(record) + "\n")
            except json.JSONDecodeError:
                print(
                    f"  [WARNING] Failed to decode JSON for problem {problem['id']}. Saving raw output."
                )
                record = {
                    "problem_id": problem["id"],
                    "problem_text": problem["text"],
                    "model_output_raw": generated_text,
                }
                f.write(json.dumps(record) + "\n")

    print(f"\nExperiment complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
