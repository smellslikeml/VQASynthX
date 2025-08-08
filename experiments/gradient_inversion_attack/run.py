import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizer
import argparse

def main(args):
    """
    Minimal experiment to demonstrate gradient inversion on a text input,
    inspired by the GRAB repository. This simulates an attack on a model
    being trained with data similar to that produced by VQASynth.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup model and tokenizer
    model_name = 'bert-base-uncased'
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval() # Ensure model is in eval mode, consistent with attack setups

    # For this attack, we only need the embedding layer's gradient.
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradients only for the embedding layer
    model.bert.embeddings.word_embeddings.weight.requires_grad = True

    # 2. Define the private data (simulating a VQA sample)
    # This would be a question or answer from the VQASynth pipeline.
    original_text = "Is the red chair to the left of the blue table?"
    print(f"Original Text: '{original_text}'")

    inputs = tokenizer(original_text, return_tensors='pt', padding='max_length', max_length=args.seq_len, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    labels = input_ids.clone()

    # 3. Compute the ground truth gradient
    model.zero_grad()
    true_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    true_loss = true_outputs.loss
    true_gradient = torch.autograd.grad(true_loss, model.bert.embeddings.word_embeddings.weight)[0]
    
    # 4. Initialize a dummy input for reconstruction
    dummy_input_embeddings = torch.randn(
        (1, args.seq_len, model.config.hidden_size),
        device=device,
        requires_grad=True
    )
    optimizer = torch.optim.Adam([dummy_input_embeddings], lr=args.lr)

    print("\nStarting gradient inversion attack...")
    for step in range(args.steps):
        optimizer.zero_grad()
        model.zero_grad()
        
        # Forward pass with dummy embeddings
        dummy_outputs = model(inputs_embeds=dummy_input_embeddings, attention_mask=attention_mask, labels=labels)
        dummy_loss = dummy_outputs.loss
        
        # Calculate the gradient for the dummy input
        dummy_gradient = torch.autograd.grad(dummy_loss, model.bert.embeddings.word_embeddings.weight, create_graph=True)[0]

        # Calculate the loss between gradients and update the dummy input
        grad_loss = F.mse_loss(dummy_gradient, true_gradient)
        grad_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step: {step:04d}, Gradient Loss: {grad_loss.item():.4f}")

    # 5. Decode the reconstructed input
    # Find the closest token for each position in the sequence
    embedding_weights = model.get_input_embeddings().weight.data
    
    # Calculate cosine similarity
    normalized_dummy_input = F.normalize(dummy_input_embeddings.squeeze(0), p=2, dim=1)
    normalized_embedding_weights = F.normalize(embedding_weights, p=2, dim=1)
    
    cosine_sim = torch.matmul(normalized_dummy_input, normalized_embedding_weights.t())
    reconstructed_ids = torch.argmax(cosine_sim, dim=1)

    reconstructed_text = tokenizer.decode(reconstructed_ids, skip_special_tokens=True)
    
    print("\n--- Attack Complete ---")
    print(f"Original Text:      '{original_text}'")
    print(f"Reconstructed Text: '{reconstructed_text}'")
    print("-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Inversion Attack Demo")
    parser.add_argument('--seq_len', type=int, default=32, help='Sequence length')
    parser.add_argument('--steps', type=int, default=1000, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for the attack')
    args = parser.parse_args()
    main(args)
