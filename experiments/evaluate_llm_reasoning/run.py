import argparse
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the models to be evaluated, inspired by RedesignAutonomy's multi-model evaluation script
MODEL_MAP = {
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b-it",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# A small, representative dataset of spatial reasoning prompts to act as our evaluation set
PROMPTS_DATA = {
    "id": [1, 2, 3],
    "prompt": [
        "I have a coffee mug to the left of my keyboard, and my mouse is to the right of my keyboard. If I move the mouse to be in front of the mug, where is the mouse relative to the keyboard?",
        "A box is on a table. A ball is inside the box. If I lift the box, what happens to the ball's position relative to the table?",
        "Imagine a room with a door on the north wall and a window on the east wall. If you are standing in the center facing the door and then turn 90 degrees clockwise, what are you facing?",
    ],
}


def create_prompts_csv(path="prompts.csv"):
    """Creates a default prompts CSV if it doesn't exist."""
    if not os.path.exists(path):
        logging.info(f"Creating default prompts file at {path}")
        df = pd.DataFrame(PROMPTS_DATA)
        df.to_csv(path, index=False)
    else:
        logging.info(f"Using existing prompts file at {path}")


def run_evaluation(model_alias, prompts_df, output_dir):
    """Runs evaluation for a single model."""
    if model_alias not in MODEL_MAP:
        logging.error(f"Model alias '{model_alias}' not found in MODEL_MAP.")
        return

    model_id = MODEL_MAP[model_alias]
    logging.info(f"--- Evaluating model: {model_alias} ({model_id}) ---")

    try:
        # Setup model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,  # Required for models like Phi-3
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        results = []
        for index, row in prompts_df.iterrows():
            prompt_text = row["prompt"]
            logging.info(f"Processing prompt ID {row['id']}: '{prompt_text[:80]}...'")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that excels at spatial reasoning.",
                },
                {"role": "user", "content": prompt_text},
            ]

            formatted_prompt = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            outputs = pipe(
                formatted_prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            # Extract only the assistant's response
            generated_text = (
                outputs[0]["generated_text"]
                .split(tokenizer.chat_template.split('{message["content"]}')[-1])[-1]
                .strip()
            )

            results.append(
                {
                    "prompt_id": row["id"],
                    "prompt": prompt_text,
                    "response": generated_text,
                }
            )

        output_path = os.path.join(output_dir, f"{model_alias}_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results for {model_alias} saved to {output_path}")

    except Exception as e:
        logging.error(
            f"Failed to evaluate model {model_alias}. Error: {e}", exc_info=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM baseline spatial reasoning."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["phi-3-mini"],
        choices=list(MODEL_MAP.keys()),
        help="A list of model aliases to evaluate.",
    )
    parser.add_argument(
        "--prompts_csv",
        type=str,
        default="prompts.csv",
        help="Path to the CSV file containing prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save evaluation results.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    create_prompts_csv(os.path.join(os.path.dirname(__file__), args.prompts_csv))

    prompts_df = pd.read_csv(os.path.join(os.path.dirname(__file__), args.prompts_csv))

    for model_alias in args.models:
        run_evaluation(model_alias, prompts_df, args.output_dir)


if __name__ == "__main__":
    main()
