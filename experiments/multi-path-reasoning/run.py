import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# PROMPT ENGINEERING SECTION
# Inspired by the need for structured reasoning as seen in llm_recovery's approach
# to decision-making.

GENERATE_PROMPT_TEMPLATE = """
[INST] You are an expert in spatial reasoning. Based on the following context about objects in an image, generate three distinct, plausible, step-by-step reasoning paths to answer the user's question. Each path should be a separate line of thought. Provide only the reasoning paths, each enclosed in <path> tags. Do not add any other commentary.

**Context:**
{context}

**Question:**
{question}
[/INST]
"""

EVALUATE_PROMPT_TEMPLATE = """
[INST] You are a meticulous evaluator of logic. Review the following reasoning paths, which all attempt to answer the question based on the provided context. Identify the single best reasoning path that is the most logical, direct, and factually consistent with the context. Output only the full text of the best path, with no extra explanation or formatting.

**Context:**
{context}

**Question:**
{question}

**Reasoning Paths to Evaluate:**
{paths_string}
[/INST]
"""


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multi-path reasoning data for VQA."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the input dataset (Hugging Face path or local JSONL file).",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split to process."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSONL file.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Hugging Face model ID for generation and evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for the generator.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional: number of samples to process from the dataset.",
    )
    return parser.parse_args()


def extract_paths(generated_text: str) -> list[str]:
    """Extracts reasoning paths from the model's raw output."""
    paths = []
    for part in generated_text.split("<path>"):
        if "</path>" in part:
            paths.append(part.split("</path>")[0].strip())
    return paths


def main():
    """Main function to run the data synthesis process."""
    args = parse_args()

    print(f"Loading model: {args.model_id}")
    # The SOURCE repo uses bitsandbytes and other optimizations. For this minimal experiment,
    # we use a simpler loading mechanism but acknowledge that for larger models, quantization
    # as seen in `llm_recovery` would be essential.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Loading dataset: {args.dataset_path}")
    if args.dataset_path.endswith(".jsonl"):
        dataset = load_dataset(
            "json", data_files=args.dataset_path, split=args.dataset_split
        )
    else:
        dataset = load_dataset(args.dataset_path, split=args.dataset_split)

    if args.sample_size:
        dataset = dataset.select(range(args.sample_size))

    print(f"Processing {len(dataset)} samples...")
    with open(args.output_file, "w") as f_out:
        for i, sample in enumerate(dataset):
            # Assumes input format has 'context' and 'question' fields.
            # This can be adapted to the specific output of the VQASynth `prompt_stage`.
            context = sample.get("context", "")
            question = sample.get("question", "")

            if not context or not question:
                print(f"Skipping sample {i} due to missing context or question.")
                continue

            # 1. Generate multiple reasoning paths (the "lookahead")
            generation_prompt = GENERATE_PROMPT_TEMPLATE.format(
                context=context, question=question
            )
            raw_generated_text = generator(generation_prompt)[0]["generated_text"]
            # The text from the pipeline includes the prompt, so we extract just the generated part.
            generated_part = raw_generated_text.split("[/INST]")[-1]
            reasoning_paths = extract_paths(generated_part)

            if len(reasoning_paths) < 2:
                print(f"Skipping sample {i}: could not generate enough distinct paths.")
                continue

            # 2. Evaluate the paths to select the best one (the "decision")
            paths_string = "\n\n".join(
                [f"Path {idx+1}:\n{path}" for idx, path in enumerate(reasoning_paths)]
            )
            evaluation_prompt = EVALUATE_PROMPT_TEMPLATE.format(
                context=context, question=question, paths_string=paths_string
            )
            raw_evaluation_text = generator(evaluation_prompt, max_new_tokens=512)[0][
                "generated_text"
            ]
            selected_reasoning = raw_evaluation_text.split("[/INST]")[-1].strip()

            # 3. Save the enriched data point
            output_record = {
                **sample,  # Keep original data
                "reasoning_paths": reasoning_paths,
                "selected_reasoning": selected_reasoning,
            }
            f_out.write(json.dumps(output_record) + "\n")
            print(f"Processed sample {i+1}/{len(dataset)}")

    print(f"Multi-path reasoning data saved to {args.output_file}")


if __name__ == "__main__":
    main()
