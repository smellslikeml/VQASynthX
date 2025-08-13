import torch
from fla.modules import FusedLinearCrossEntropy

# This script demonstrates how to use FusedLinearCrossEntropy from the flash-linear-attention
# library to create a memory-efficient language model head for training.

# --- Configuration ---
# These would come from your model and data configuration
VOCAB_SIZE = 32000
HIDDEN_SIZE = 4096
BATCH_SIZE = 4
SEQ_LEN = 2048

# Check for CUDA availability for the demo
if not torch.cuda.is_available():
    print("CUDA not available. This experiment requires a GPU.")
    exit()

device = "cuda"
dtype = torch.bfloat16

# --- Mock Model & Data ---
# In a real scenario, `mock_hidden_states` would be the output of your VLM's backbone.
# Shape: (batch_size, seq_len, hidden_size)
mock_hidden_states = torch.randn(
    BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype
)

# Target labels for the language modeling task
# Shape: (batch_size, seq_len)
mock_labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

print(
    f"Created mock data:\n- Hidden states: {mock_hidden_states.shape}\n- Labels: {mock_labels.shape}\n"
)

# --- Integration of FusedLinearCrossEntropy ---
# 1. Instantiate the fused layer. It replaces the final `nn.Linear` projection (i.e., the `lm_head`).
# The layer must be on the correct device and use the appropriate dtype.
fused_ce_loss_layer = FusedLinearCrossEntropy(
    input_size=HIDDEN_SIZE,
    output_size=VOCAB_SIZE,
    bias=False,  # `lm_head` layers often have no bias
    device=device,
    dtype=dtype,
)
print("Instantiated `fla.modules.FusedLinearCrossEntropy` layer:", fused_ce_loss_layer)

# --- Training Step Simulation ---
# In a standard training loop, you would perform the following steps.

# 2. Compute loss.
# The layer takes the last hidden state and the labels as input.
# It computes `cross_entropy(linear(hidden_states), labels)` in a single fused kernel.
# The output is the scalar loss tensor.
print("\n--- Simulating a training step ---")
loss = fused_ce_loss_layer(mock_hidden_states, mock_labels)

# 3. Backpropagate.
# The fused kernel also implements the backward pass.
loss.backward()

# --- Verification ---
print(f"Successfully computed loss: {loss.item():.4f}")

# Check if the gradient was computed for the layer's weight tensor.
grad_is_present = fused_ce_loss_layer.weight.grad is not None
print(f"Gradient has been computed for the layer's weight: {grad_is_present}")
if grad_is_present:
    print(f"Gradient shape: {fused_ce_loss_layer.weight.grad.shape}")


# --- Notes for full implementation in `experimental-vqasynth` ---
#
# To integrate this into the `llava_train` stage:
#
# 1. ADD DEPENDENCY:
#    Add `flash-linear-attention` to `requirements.txt` or the training Dockerfile.
#    `pip install flash-linear-attention`
#
# 2. MODIFY MODEL ARCHITECTURE:
#    - In your model loading script, identify the language model head (e.g., `model.lm_head`).
#    - Replace this `nn.Linear` layer with an instance of `fla.modules.FusedLinearCrossEntropy`:
#      `model.lm_head = FusedLinearCrossEntropy(input_size=model.config.hidden_size, output_size=model.config.vocab_size, ...)`
#
# 3. ADAPT TRAINING LOGIC:
#    - The key change is to prevent the model's default forward pass from calculating the loss.
#    - Instead, you call the new `lm_head` (our fused layer) manually.
#
#    Example modification in a typical training step:
#    ```python
#    # inputs = batch.to(device)
#    # labels = inputs.pop('labels') # Remove labels from model inputs
#
#    # # Forward pass through the model backbone *without* internal loss calculation
#    # outputs = model(**inputs)
#    # hidden_states = outputs.last_hidden_state
#
#    # # Calculate loss using the fused layer, which is now the lm_head
#    # loss = model.lm_head(hidden_states, labels)
#    #
#    # loss.backward()
#    # optimizer.step()
#    ```
#    This avoids the memory overhead of storing the full logit tensor.
