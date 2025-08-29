import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Based on the OverFill repository, we use a large model for the prompt processing (prefill)
# and a smaller, pruned model for the token generation (decoding).
# See: https://github.com/friendshipkim/overfill
PREFILL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DECODE_MODEL_ID = (
    "friendshipkim/overfill-Llama-3.1-8B-Instruct-pruned-h0.43-i0.43-a0.0-d0.0-bf16"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float32
)


def main():
    """
    Demonstrates the two-stage decoding process from OverFill.
    1. A large "prefill" model processes the initial prompt to generate a KV cache.
    2. A smaller "decode" model uses this initial KV cache to generate subsequent tokens autoregressively.
    This optimizes inference by using a faster, smaller model for the sequential generation part.
    """
    print(f"Using device: {DEVICE} with dtype: {DTYPE}")

    # --- 1. Load Models and Tokenizer ---
    print("Loading models and tokenizer...")
    # A Hugging Face token with access to Llama 3 is required.
    tokenizer = AutoTokenizer.from_pretrained(PREFILL_MODEL_ID)

    # Load the large model for the prefill stage
    prefill_model = AutoModelForCausalLM.from_pretrained(
        PREFILL_MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )

    # Load the smaller, pruned model for the decoding stage
    decode_model = AutoModelForCausalLM.from_pretrained(
        DECODE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    print("Models and tokenizer loaded.")

    # --- 2. Prepare Input ---
    # This prompt simulates a complex spatial reasoning query that VQASynth might generate.
    prompt = (
        "Based on the scene analysis, a red forklift is located approximately 5 meters to the "
        "left of a stack of brown cardboard boxes. A man in a red hat is walking near a wooden pallet, "
        "about 2 meters away. The entire scene is in a large warehouse with concrete floors. "
        "Question: Does the red forklift in the warehouse appear on the left side of the brown "
        "cardboard boxes stacked? Provide a step-by-step reasoning for your answer."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions about spatial relationships in an image.",
        },
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)

    prompt_length = input_ids.shape[-1]
    print(f"\nPrompt length: {prompt_length} tokens")

    # --- 3. Prefill Stage (using the large model) ---
    print("\n--- Starting Prefill Stage (Large Model) ---")
    start_time = time.time()
    with torch.no_grad():
        # The prefill model processes the entire prompt in parallel to generate the initial KV cache.
        prefill_outputs = prefill_model(input_ids, use_cache=True)
        past_key_values = prefill_outputs.past_key_values
        # The logits for the *first* generated token are taken from the large model.
        next_token_logits = prefill_outputs.logits[:, -1, :]

    prefill_duration = time.time() - start_time
    print(f"Prefill stage completed in {prefill_duration:.4f} seconds.")

    # --- 4. Decode Stage (using the small model) ---
    print("\n--- Starting Decode Stage (Small Model) ---")
    generated_ids = []
    max_new_tokens = 100

    start_time = time.time()

    # Greedily select the first token based on the prefill model's logits
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    generated_ids.append(next_token.item())

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            # The small decode model takes the *single* next token and the KV cache.
            # This is much faster than processing the full sequence again.
            decode_outputs = decode_model(
                input_ids=next_token, past_key_values=past_key_values, use_cache=True
            )

        # Update the KV cache and get the logits for the next token
        past_key_values = decode_outputs.past_key_values
        next_token_logits = decode_outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            print("EOS token generated. Stopping.")
            break

        generated_ids.append(next_token.item())

    decode_duration = time.time() - start_time
    print(
        f"Decode stage completed in {decode_duration:.4f} seconds for {len(generated_ids)} tokens."
    )

    # --- 5. Print Results ---
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n--- Full Generated Response ---")
    print(generated_text)
    print("\n--- Experiment Summary ---")
    print(f"Total time: {prefill_duration + decode_duration:.4f} seconds")
    print(
        f"Tokens per second (decode only): {len(generated_ids) / decode_duration:.2f} t/s"
    )


if __name__ == "__main__":
    main()
