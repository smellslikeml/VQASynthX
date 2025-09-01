import torch
import torch.optim as optim
from transformers import AutoProcessor, LlavaLlamaForCausalLM
import argparse
import os


def run_unlearning(args):
    """Applies one-shot unlearning to a fine-tuned VLM based on the IAU method."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the fine-tuned model and processor
    print(f"Loading base model from {args.model_path}...")
    model = LlavaLlamaForCausalLM.from_pretrained(args.model_path).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 2. Prepare data loaders (using dummy data for this self-contained example)
    # In a real pipeline, these would load specific samples identified for unlearning/retention.
    # The key idea from IAU is to use single, large batches for the unlearning step.

    # --- Data to Retain ---
    # A small, representative set of 'good' data from the original finetuning set.
    # This ensures the model doesn't forget its primary task.
    retain_texts = [
        "USER: <image>\nWhat is in this image? ASSISTANT: A photo of a city skyline at dusk."
    ]
    # Use a dummy image tensor. In practice, this would be loaded from a file.
    retain_images = [torch.rand(3, 336, 336, device=device)]
    retain_inputs = processor(
        text=retain_texts, images=retain_images, return_tensors="pt", padding=True
    ).to(device)

    # --- Data to Forget ---
    # The specific 'bad' samples we want the model to unlearn (e.g., a sample containing PII).
    forget_texts = [
        "USER: <image>\nWhat is the name of the person in the blue shirt? ASSISTANT: The person's name is John Doe."
    ]
    # Dummy image for the sample to be forgotten.
    forget_images = [torch.rand(3, 336, 336, device=device)]
    forget_inputs = processor(
        text=forget_texts, images=forget_images, return_tensors="pt", padding=True
    ).to(device)

    # 3. Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Perform the unlearning step
    model.train()
    optimizer.zero_grad()

    # --- RETAIN GRADIENT ---
    # Calculate loss on the data to retain and backpropagate normally.
    # This gradient descent step pushes the model to keep what it learned from good data.
    retain_outputs = model(**retain_inputs, labels=retain_inputs["input_ids"])
    retain_loss = retain_outputs.loss
    print(f"Retain loss: {retain_loss.item():.4f}")
    retain_loss.backward()

    # --- FORGET GRADIENT ---
    # Calculate loss on the data to forget and backpropagate the *negative* loss.
    # This is gradient ascent, pushing the model away from what it learned from bad data.
    # This is the core technique from the IAU paper.
    forget_outputs = model(**forget_inputs, labels=forget_inputs["input_ids"])
    forget_loss = -args.alpha * forget_outputs.loss
    print(f"Forget loss (negated and weighted): {forget_loss.item():.4f}")
    forget_loss.backward()

    # 5. Apply the combined gradients from both steps
    print("Applying combined gradients...")
    optimizer.step()

    # 6. Save the unlearned model
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(f"Unlearning complete. Saving model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)
    print("Done.")


def create_dummy_model_for_testing(path="dummy_vlm"):
    """Creates a minimal LLaVA model for testing the script without real weights."""
    if os.path.exists(path):
        return
    print(f"Creating dummy model for testing at ./{path}")
    from transformers import LlavaConfig, LlamaConfig, CLIPVisionConfig

    # Using a tiny config for a model that can be created quickly in memory
    text_config = LlamaConfig(
        num_hidden_layers=1, hidden_size=32, intermediate_size=64, num_attention_heads=2
    )
    vision_config = CLIPVisionConfig(
        num_hidden_layers=1,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        image_size=30,
    )
    config = LlavaConfig(text_config=text_config, vision_config=vision_config)

    model = LlavaLlamaForCausalLM(config)
    # Use a real processor for its tokenizer and config, which is lightweight
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    model.save_pretrained(path)
    processor.save_pretrained(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Efficiently unlearn samples from a fine-tuned VLM."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dummy_vlm",
        help="Path to the VLM to unlearn from.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="unlearned_vlm",
        help="Path to save the unlearned model.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-7, help="Learning rate for the unlearning step."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weighting factor for the forget loss, as in the IAU paper.",
    )

    args = parser.parse_args()

    # Create a dummy model if one doesn't exist, so the script is runnable out-of-the-box
    create_dummy_model_for_testing(args.model_path)

    run_unlearning(args)
