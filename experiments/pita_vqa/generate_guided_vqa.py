# Adapted from SOURCE: imdb_gen/eval_ckpt.py
# This script performs PITA-guided generation for VQA prompts.

import argparse
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from experiments.pita_vqa.classifier import CustomLlamaForClassification


def format_prompt(question):
    # Prompt template for generation
    return f"USER: <image>\n{question} ASSISTANT:"


def generate_guided(
    model,
    classifier_model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    eta=1.0,
):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits from the base model
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]

            # Get reward/score from the classifier model
            classifier_logits = classifier_model(generated_ids)
            reward = classifier_logits[-1].item()

            # PITA guidance: modify logits based on reward gradient approximation
            # For simplicity in this adaptation, we use a simplified guidance logic.
            # The core idea is to blend original logits with a preference score.
            # A more faithful implementation would use classifier's gradients.

            # Here, we simulate the effect by adding a scaled reward to the logits.
            # A positive reward should 'encourage' the current path.
            # This is a simplification; the original paper uses a more direct lookahead.

            # In PITA, we'd compute grad of reward w.r.t logits.
            # grad = torch.autograd.grad(outputs=reward, inputs=next_token_logits)[0]
            # For this minimal PoC, we'll just use the reward to scale logits slightly.
            # This is not a faithful impl, but demonstrates the control flow.

            # Let's stick to the core generation logic for clarity of the proposal.
            # The key is *how* the classifier is called in the loop.

            # The PITA paper uses classifier log-probabilities to adjust next-token logits.
            # q(y_t | y_<t) propto p_ref(y_t | y_<t) * exp(eta * r(y_<=t))
            # We approximate r(y_<=t) with the classifier's output.

            log_probs = F.log_softmax(next_token_logits, dim=-1)
            guided_log_probs = log_probs + eta * reward

            # Sample from the modified distribution
            if temperature > 0:
                probs = F.softmax(guided_log_probs / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(guided_log_probs, dim=-1).unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(
        generated_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--classifier_ckpt_path", type=str, required=True)
    parser.add_argument(
        "--prompts_path",
        type=str,
        required=True,
        help="JSONL file with 'question' field",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0, help="Guidance strength")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for generation
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id).to(device)
    base_model.eval()

    # Load PITA classifier model
    classifier_model = CustomLlamaForClassification(args.base_model_id).to(device)
    classifier_model.load_state_dict(
        torch.load(os.path.join(args.classifier_ckpt_path, "pytorch_model.bin"))
    )
    classifier_model.eval()

    prompts = []
    with open(args.prompts_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "generated_vqa_responses.jsonl")

    print(f"Generating guided responses on {device}...")
    with open(output_file, "w") as f_out:
        for item in tqdm(prompts, desc="Generating"):
            question = item["question"]
            prompt_text = format_prompt(question)

            generated_text = generate_guided(
                model=base_model,
                classifier_model=classifier_model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eta=args.eta,
            )

            result = {"question": question, "guided_answer": generated_text}
            f_out.write(json.dumps(result) + "\n")

    print(f"Guided generation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
