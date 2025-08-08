import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

# This experiment integrates the data generation philosophy from open-r1.
# The open-r1 project focuses on distilling reasoning traces from powerful models
# to create datasets like 'Mixture-of-Thoughts'. See `src/open_r1/generate.py`.
# We apply the same principle here: use a strong reasoning model to generate
# a chain-of-thought for the spatial VQA prompts created in the VQASynth pipeline.

def generate_reasoning_trace(llm, prompt, sampling_params):
    """Generates a reasoning trace for a given prompt."""
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()

def main(input_dir, output_dir, model_id, max_samples=None):
    """
    Loads VQA prompts, generates a CoT reasoning trace using an open-r1 style model,
    and saves the augmented dataset.
    """
    print(f"Loading model: {model_id}")
    # Using vLLM as suggested in the open-r1 installation guide for efficient inference.
    llm = LLM(model=model_id, trust_remote_code=True, tensor_parallel_size=1)
    
    # These sampling parameters are a reasonable starting point.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1024)

    input_path = Path(input_dir) / "prompts.jsonl"
    output_path = Path(output_dir) / "reasoning.jsonl"
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading prompts from {input_path}")
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]

    if max_samples:
        data = data[:max_samples]
        print(f"Processing a maximum of {max_samples} samples.")

    results = []
    for item in tqdm(data, desc="Generating Reasoning Traces"):
        # The prompt format is designed to elicit a step-by-step thought process,
        # inspired by the datasets mentioned in the open-r1 README.
        question = item['question']
        prompt = f"<|im_start|>system\nYou are an expert in spatial reasoning. Analyze the user's question and provide a step-by-step thinking process that leads to the final answer. Think out loud.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        reasoning_trace = generate_reasoning_trace(llm, prompt, sampling_params)
        
        new_item = item.copy()
        new_item['reasoning_trace'] = reasoning_trace
        results.append(new_item)

    print(f"Writing augmented data to {output_path}")
    with open(output_path, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print("Reasoning generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chain-of-Thought reasoning for VQA data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input prompts.jsonl file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output reasoning.jsonl file.")
    parser.add_argument("--model_id", type=str, default="open-r1/OpenR1-Distill-7B", help="Hugging Face model ID to use for generation.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process for testing.")
    
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_id, args.max_samples)
