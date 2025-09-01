import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# This script is a minimal implementation of the 'echo generation' technique
# described in the LogicORM paper (SOURCE repo).
# The goal is to generate flawed reasoning chains by providing the model
# with a deliberately incorrect conclusion and prompting it to justify that conclusion.
# This creates negative training data for training a reward model or for preference tuning.


def generate_echo_reasoning(model, tokenizer, premise, incorrect_conclusion, device):
    """
    Generates a flawed reasoning chain by 'echoing' an incorrect conclusion.

    Args:
        model: The pretrained Hugging Face model.
        tokenizer: The tokenizer for the model.
        premise (str): The logical premises.
        incorrect_conclusion (str): The incorrect conclusion to steer the model towards.
        device (str): The device to run inference on ('cuda', 'cpu', etc.).

    Returns:
        str: The generated flawed reasoning chain.
    """
    # The prompt is designed to coax the model into producing a flawed argument.
    # It explicitly asks the model to justify a known incorrect outcome, simulating the
    # 'echo' effect where the LLM reflects the prompt's assumptions.
    # This is inspired by the data generation process in `src/data_generation.py` of the SOURCE repo.
    prompt_template = (
        "You are an expert in logical fallacies. Your task is to construct a plausible-sounding but ultimately incorrect step-by-step argument. "
        "Start from the given premises and forcibly arrive at the given incorrect conclusion.\n\n"
        "## Premises: {premise}\n"
        "## Incorrect Conclusion: {incorrect_conclusion}\n\n"
        "## Flawed Reasoning Chain:"
    )

    formatted_prompt = prompt_template.format(
        premise=premise, incorrect_conclusion=incorrect_conclusion
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that follows instructions precisely, even if they lead to logically incorrect outcomes.",
        },
        {"role": "user", "content": formatted_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    input_length = model_inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_length:]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Generate flawed reasoning chains using the LogicORM echo technique."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="The Hugging Face model to use for generation.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )

    # Example of a simple deductive reasoning problem.
    # The correct conclusion is 'Socrates is mortal.'
    premise = "1. All men are mortal.\n2. Socrates is a man."
    incorrect_conclusion = "Socrates is not mortal."

    print("-" * 50)
    print(f"Premise:\n{premise}")
    print(f"Forcing Incorrect Conclusion:\n{incorrect_conclusion}")
    print("-" * 50)
    print("Generated Flawed Reasoning (Echo Generation):\n")

    flawed_reasoning = generate_echo_reasoning(
        model, tokenizer, premise, incorrect_conclusion, device
    )
    print(flawed_reasoning)


if __name__ == "__main__":
    main()
