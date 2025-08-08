import torch
import torch.nn as nn
import math


class TenVOOLayer(nn.Module):
    """
    A simplified implementation of a LoRA-like layer for Conv3d, inspired by TenVOO's
    parameter-efficient approach. This layer freezes the original Conv3d weights
    and introduces a small, trainable parallel branch using low-rank decomposition.
    """
    def __init__(self, original_conv: nn.Conv3d, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding

        # Freeze the original convolution layer
        self.original_conv = original_conv
        self.original_conv.requires_grad_(False)

        # Low-rank decomposition path, mimicking a LoRA-style update
        # 1. A pointwise conv to reduce channel dimension to the rank
        # 2. A standard conv to expand from the rank back to the output channels
        self.lora_A = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.rank,
            kernel_size=(1, 1, 1), # Pointwise convolution for channel projection
            stride=1,
            padding=0,
            bias=False
        )
        self.lora_B = nn.Conv3d(
            in_channels=self.rank,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )

        # Initialize the low-rank matrices
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Original path (frozen)
        original_output = self.original_conv(x)
        # Low-rank adapter path (trainable)
        lora_output = self.lora_B(self.lora_A(x))
        return original_output + lora_output

def create_simple_3d_model():
    """Creates a simple 3D CNN model for demonstration."""
    return nn.Sequential(
        nn.Conv3d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv3d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv3d(32, 16, kernel_size=3, padding=1)
    )

def apply_tenvoo_to_model(model: nn.Module, rank: int = 4):
    """
    Recursively traverses the model and replaces all Conv3d layers
    with TenVOOLayer wrappers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv3d):
            setattr(model, name, TenVOOLayer(module, rank=rank))
        else:
            # Recurse into sequential and other container modules
            apply_tenvoo_to_model(module, rank=rank)

def count_parameters(model: nn.Module, trainable_only: bool = False):
    """Counts total or trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)

if __name__ == "__main__":
    print("--- TenVOO PEFT for 3D Models Experiment ---")
    
    # 1. Create the base model
    model = create_simple_3d_model()
    original_params = count_parameters(model)
    print(f"Original model created. Total parameters: {original_params:,}")

    # 2. Apply TenVOO-inspired PEFT
    print("\nApplying TenVOO-inspired adapter with rank=4...")
    apply_tenvoo_to_model(model, rank=4)
    print("Model adapted with TenVOO layers.")

    # 3. Compare parameter counts
    total_params_after_peft = count_parameters(model, trainable_only=False)
    trainable_peft_params = count_parameters(model, trainable_only=True)
    
    print(f"\n--- Parameter Comparison ---")
    print(f"Total original model parameters: {original_params:,}")
    print(f"Total parameters in adapted model (frozen base + adapter): {total_params_after_peft:,}")
    print(f"Trainable TenVOO adapter parameters: {trainable_peft_params:,}")

    reduction = 1 - (trainable_peft_params / original_params)
    print(f"\nParameter reduction for training: {reduction:.2%}")

    print("\nThis demonstrates that TenVOO allows fine-tuning with a small fraction")
    print("of the original model's parameters, enabling efficient adaptation of large 3D models.")
    
    # --- Mock training step to show it works ---
    print("\n--- Mock forward pass ---")
    try:
        dummy_input = torch.randn(1, 1, 32, 32, 32) # (N, C, D, H, W)
        output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        # Mock backward pass
        loss = output.sum()
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"An error occurred during the mock run: {e}")
