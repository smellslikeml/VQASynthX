import os
import json
import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_entropy(logits):
    """Calculates the Shannon entropy for a given logit tensor."""
    # logits shape: (batch_size, sequence_length, vocab_size)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def process_and_filter(input_file, output_file, model_name, entropy_threshold, device):
    """
    Processes VQA pairs, calculates the entropy of the first generated token
    of the answer, and filters out samples above a given entropy threshold.
    """
    logging.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    filtered_samples = []
    total_samples = 0
    filtered_out_count = 0

    with open(input_file, "r") as f_in:
        lines = f_in.readlines()
        total_samples = len(lines)
        logging.info(f"Found {total_samples} samples in {input_file}")

        for line in tqdm(lines, desc="Analyzing sample uncertainty"):
            try:
                data = json.loads(line)
                # Assumes LLaVA-style conversation format: [{"from": "human", ...}, {"from": "gpt", ...}]
                prompt = data["conversations"][0]["value"]
                answer = data["conversations"][1]["value"]

                # Tokenize prompt and answer to find the boundary
                prompt_tokens = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=True
                ).to(device)

                # We need the full input to reproduce the logits that generated the answer
                full_text = prompt + answer
                full_input_ids = (
                    tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
                    .to(device)
                    .input_ids
                )

                with torch.no_grad():
                    outputs = model(input_ids=full_input_ids)
                    logits = outputs.logits

                # The logit for predicting token `i` is at `logits[:, i-1, :]`.
                # We want the logits for the first token of the answer.
                prompt_len = prompt_tokens.input_ids.shape[1]
                answer_logits = logits[
                    :, prompt_len - 1 : -1, :
                ]  # Logits that predict each token of the answer

                if answer_logits.shape[1] == 0:
                    logging.warning(
                        "Skipping sample with empty answer after tokenization."
                    )
                    continue

                # The core idea from RAGTruth_Xtended: focus on the first token's signal
                first_token_logits = answer_logits[:, 0, :].unsqueeze(
                    1
                )  # Keep dimensions for entropy function
                first_token_entropy = calculate_entropy(first_token_logits)[0, 0].item()

                if first_token_entropy <= entropy_threshold:
                    filtered_samples.append(data)
                else:
                    filtered_out_count += 1

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logging.warning(
                    f"Skipping malformed line or data structure: {e}. Line: {line.strip()}"
                )
                continue

    logging.info(
        f"Finished processing. Total samples: {total_samples}. Filtered out: {filtered_out_count}. Kept: {len(filtered_samples)}."
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f_out:
        for sample in filtered_samples:
            f_out.write(json.dumps(sample) + "\n")

    logging.info(f"Filtered dataset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter VQA data based on first-token entropy of the answer."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the filtered JSONL file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Hugging Face model to use for logit reproduction.",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=1.5,
        help="Entropy threshold for filtering. Samples with higher first-token entropy will be discarded.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    process_and_filter(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        entropy_threshold=args.entropy_threshold,
        device=device,
    )


if __name__ == "__main__":
    main()
