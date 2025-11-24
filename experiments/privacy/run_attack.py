import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# --- Configuration ---
# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Experiment parameters
MODEL_NAME = "gpt2"  # Using GPT2 as it's a standard causal LM, simple to attack.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ORIGINAL_TEXT = "What is the distance between the red forklift and the cardboard boxes?"
SEQUENCE_LENGTH = 16  # Shorter sequence is easier to reconstruct
LEARNING_RATE = 0.05
ATTACK_STEPS = 500
LOG_INTERVAL = 100


def get_model_and_tokenizer():
    """Loads the model and tokenizer."""
    logging.info(f"Loading model and tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()  # Set to eval mode, as we are not training the model itself
    logging.info(f"Model and tokenizer loaded, running on {DEVICE}.")
    return model, tokenizer


def get_real_gradients(model, tokenizer):
    """
    Simulates a single training step on the original text and captures the gradients
    with respect to the model's parameters. These are the "leaked" gradients.
    """
    logging.info("--- Step 1: Simulating user training to get real gradients ---")

    # Tokenize input
    inputs = tokenizer(
        ORIGINAL_TEXT,
        return_tensors="pt",
        max_length=SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    labels = input_ids.clone()

    # Ensure model is ready for gradient calculation
    model.zero_grad()

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logging.info(f"Original text loss: {loss.item():.4f}")

    # Backward pass to calculate gradients
    loss.backward()

    # Store the real gradients (we'll only use the embedding layer's gradient for simplicity)
    embedding_layer = model.get_input_embeddings()
    real_gradients = embedding_layer.weight.grad.clone().detach()

    model.zero_grad()  # Clean up gradients
    return real_gradients


def run_gradient_inversion_attack(model, tokenizer, real_gradients):
    """
    Performs a gradient matching attack to reconstruct the input text.
    This is inspired by DLG and the continuous optimization part of GRAB.
    """
    logging.info("\n--- Step 2: Running Gradient Inversion Attack ---")

    # Initialize dummy inputs (random embeddings)
    embedding_layer = model.get_input_embeddings()
    dummy_input_embeddings = torch.randn(
        (1, SEQUENCE_LENGTH, embedding_layer.weight.shape[1]),
        device=DEVICE,
        requires_grad=True,
    )

    # Setup optimizer to tune the dummy embeddings
    optimizer = torch.optim.Adam([dummy_input_embeddings], lr=LEARNING_RATE)

    for step in range(ATTACK_STEPS):
        optimizer.zero_grad()
        model.zero_grad()

        # We need a loss to backpropagate from our dummy input.
        # A simple loss is the sum of the logits, which ensures gradients are computed.
        # The labels can be arbitrary as they don't affect the gradient w.r.t. parameters
        # when we use autograd.grad on an intermediate output (like the loss used here).
        dummy_outputs = model(inputs_embeds=dummy_input_embeddings)
        dummy_loss_for_grad_calc = dummy_outputs.logits.sum()

        # Calculate gradients from this dummy loss w.r.t. the embedding layer weights
        dummy_gradients = torch.autograd.grad(
            dummy_loss_for_grad_calc, [embedding_layer.weight], create_graph=True
        )[0]

        # The attack's objective is to minimize the distance between real and dummy gradients
        attack_loss = ((real_gradients - dummy_gradients) ** 2).sum()

        # Backpropagate the attack loss to update the dummy embeddings
        attack_loss.backward()

        # Update the dummy embeddings
        optimizer.step()

        if step % LOG_INTERVAL == 0 or step == ATTACK_STEPS - 1:
            logging.info(
                f"Step {step+1}/{ATTACK_STEPS} | Attack Loss: {attack_loss.item():.4f}"
            )

    logging.info(
        "--- Attack finished. Reconstructing text from optimized embeddings. ---"
    )

    # Decode the optimized embeddings back to text
    with torch.no_grad():
        # Find the closest token embedding in the vocabulary for each position
        distances = torch.cdist(
            dummy_input_embeddings.squeeze(0), embedding_layer.weight
        )
        reconstructed_ids = torch.argmin(distances, dim=1)
        reconstructed_text = tokenizer.decode(
            reconstructed_ids, skip_special_tokens=True
        )

    return reconstructed_text


def main():
    """Main function to run the demonstration."""
    model, tokenizer = get_model_and_tokenizer()
    real_gradients = get_real_gradients(model, tokenizer)
    reconstructed_text = run_gradient_inversion_attack(model, tokenizer, real_gradients)

    print("\n" + "=" * 60)
    print("                      Attack Results")
    print("=" * 60)
    print(f"Original Text:      '{ORIGINAL_TEXT}'")
    print(f"Reconstructed Text: '{reconstructed_text}'")
    print("=" * 60)
    print("Note: Perfect reconstruction is unlikely. The goal is to show that")
    print("gradients leak information about the training data.")


if __name__ == "__main__":
    main()
