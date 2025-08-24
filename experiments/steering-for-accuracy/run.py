import torch
from transformer_lens import HookedTransformer
import gc

# Configuration
MODEL_NAME = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TO_STEER = 20  # Mid-to-late layers are often good for abstract concepts
STEERING_MULTIPLIER = 1.5  # How strongly to apply the steering vector


def clear_gpu_memory():
    """Clear GPU memory cache for stability."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_mean_activations(model, texts, layer):
    """
    Get the mean residual stream activations for a list of texts at a specific layer.
    We focus on the activation at the final token position.
    """
    all_activations = []

    # Process in batches to avoid OOM errors
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = model.to_tokens(batch_texts, prepend_bos=True).to(DEVICE)

        # The hook name for the residual stream at a specific layer
        hook_name = f"blocks.{layer}.hook_resid_post"

        # We use a cache to store the activations
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

        # Get the activation at the last token position for each text in the batch
        # shape: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        activations = cache[hook_name][:, -1, :]
        all_activations.append(activations.detach())

    # Concatenate all batch activations and compute the mean
    mean_activations = torch.cat(all_activations, dim=0).mean(dim=0)
    return mean_activations


def generate_with_steering(
    model, prompt, steering_vector, layer, multiplier, max_new_tokens=10
):
    """
    Generate text while applying a steering vector to a specific layer's activations.
    """
    # Tokenize the prompt
    input_tokens = model.to_tokens(prompt, prepend_bos=True).to(DEVICE)

    # Define the hook function to add the steering vector
    def steering_hook(resid_post, hook):
        # We only apply the steering vector at the last token position during generation
        if resid_post.shape[1] >= input_tokens.shape[1]:
            resid_post[:, -1, :] += steering_vector * multiplier
        return resid_post

    hook_name = f"blocks.{layer}.hook_resid_post"

    # Generate text with the hook
    output_tokens = model.generate(
        input_tokens,
        max_new_tokens=max_new_tokens,
        fwd_hooks=[(hook_name, steering_hook)],
    )

    return model.to_string(output_tokens[0])


def main():
    """
    Main experiment script to compute a steering vector for quantitative accuracy
    and demonstrate its effect.
    """
    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}...")
    # We don't need gradients for this
    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    # --- 1. Define Datasets for Positive (Accurate) and Negative (Inaccurate) Concepts ---
    # These are simple examples to teach the model the "concept" of a correct calculation.
    # The structure "Q: ... A: ..." helps the model understand the task.
    accurate_texts = [
        "Q: What is 2 + 2? A: 4",
        "Q: How many days are in a week? A: 7",
        "Q: What is 10 - 3? A: 7",
        "Q: A triangle has how many sides? A: 3",
        "Q: What is 5 times 2? A: 10",
        "Q: How many fingers on one hand? A: 5",
    ]

    inaccurate_texts = [
        "Q: What is 2 + 2? A: 5",
        "Q: How many days are in a week? A: 8",
        "Q: What is 10 - 3? A: 6",
        "Q: A triangle has how many sides? A: 4",
        "Q: What is 5 times 2? A: 11",
        "Q: How many fingers on one hand? A: 6",
    ]

    # --- 2. Compute the Steering Vector ---
    print(f"\nComputing steering vector at layer {LAYER_TO_STEER}...")

    # Get mean activations for both concepts
    mean_accurate_activations = get_mean_activations(
        model, accurate_texts, LAYER_TO_STEER
    )
    mean_inaccurate_activations = get_mean_activations(
        model, inaccurate_texts, LAYER_TO_STEER
    )

    # The steering vector is the difference between the target concept and the undesired one
    steering_vector = mean_accurate_activations - mean_inaccurate_activations

    # Normalize the vector for more stable application
    steering_vector = steering_vector / steering_vector.norm()

    print("Steering vector computed successfully.")
    clear_gpu_memory()

    # --- 3. Test the Steering Vector's Effect ---
    test_prompt = "Q: What is 8 + 3? A:"
    print(f"\n--- Testing Generation ---")
    print(f"Prompt: '{test_prompt}'")
    print(f"Steering Layer: {LAYER_TO_STEER}, Multiplier: {STEERING_MULTIPLIER}")

    # Generate without steering (baseline)
    print("\n1. Baseline Generation (no steering):")
    unsteered_output = model.generate(test_prompt, max_new_tokens=5, temperature=0.7)
    print(f"   Output: {unsteered_output}")

    # Generate with steering
    print("\n2. Steered Generation:")
    steered_output = generate_with_steering(
        model,
        test_prompt,
        steering_vector,
        layer=LAYER_TO_STEER,
        multiplier=STEERING_MULTIPLIER,
        max_new_tokens=5,
    )
    print(f"   Output: {steered_output}")

    print("\n--- Experiment Complete ---")
    print(
        "Compare the outputs. The steered version is hypothesized to be more likely to produce the correct answer '11'."
    )


if __name__ == "__main__":
    main()
