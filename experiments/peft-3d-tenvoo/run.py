import torch
import torch.nn as nn
import re
import math
from collections import OrderedDict
from dataclasses import dataclass, field

# Try to import monai, provide helpful error message if not installed.
try:
    from monai.networks.nets import UNet
except ImportError:
    raise ImportError(
        "MONAI is not installed. Please install it to run this experiment: "
        "`pip install monai`"
    )

# ============== TENVOO IMPLEMENTATION FROM SOURCE REPO ============== #
# This section contains a minimal, self-contained implementation of TenVOO
# adapted from the xiaovhua/tenvoo repository.

TENVOO_LIST = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 1, 4),
    (1, 1, 8),
    (1, 2, 2),
    (1, 2, 4),
    (1, 2, 8),
    (2, 2, 2),
    (2, 2, 4),
]


@dataclass
class PEFTConfig:
    peft_type: str = field(default="TENVOO", metadata={"help": "The PEFT type"})


@dataclass
class TenVOOConfig(PEFTConfig):
    d_in: int = field(
        default=3, metadata={"help": "Number of dimensions for input decomposition"}
    )
    d_out: int = field(
        default=3, metadata={"help": "Number of dimensions for output decomposition"}
    )
    per_dim_list: list = field(
        default_factory=lambda: TENVOO_LIST,
        metadata={"help": "List of per-dimension options"},
    )
    target_modules: list = field(
        default_factory=list, metadata={"help": "List of module names to apply PEFT to"}
    )
    model_mode: str = field(
        default="l", metadata={"help": "TenVOO mode, 'l' for linear, 'q' for quadratic"}
    )
    rank: int = field(default=4, metadata={"help": "LoRA rank"})


class TenVOOLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        per_dim_list,
        model_mode,
        rank,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_dim_list = per_dim_list
        self.model_mode = model_mode
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.scaling = rank**-0.5
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.normal_(self.lora_A, std=1 / rank)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        lora_w = self.lora_B @ self.lora_A
        lora_w = (
            lora_w.view(self.out_features, self.in_features, 1, 1, 1) * self.scaling
        )

        output = nn.functional.conv3d(
            x, lora_w.to(x.dtype), stride=self.stride, padding=self.padding
        )
        return output.to(previous_dtype)


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def peft2nnmodel(model):
    for name, module in model.named_modules():
        if isinstance(module, TenVOOLayer):
            key = name.replace(".lora", "")
            parent, target_layer, target_name = _get_submodules(model, key)

            lora_w = module.lora_B @ module.lora_A
            lora_w = (
                lora_w.view(module.out_features, module.in_features, 1, 1, 1)
                * module.scaling
            )

            merged_w = target_layer.weight.data + lora_w.to(target_layer.weight.dtype)
            target_layer.weight.data.copy_(merged_w)

            # remove lora
            parent.lora = nn.Identity()

    return model


class TenVOOModel(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self._find_and_replace()

    def _find_and_replace(self):
        for name, module in self.model.named_modules():
            if not any(re.match(f".*{p}.*", name) for p in self.config.target_modules):
                continue

            if isinstance(module, nn.Conv3d):
                parent, _, target_name = _get_submodules(self.model, name)
                lora_layer = TenVOOLayer(
                    in_features=module.in_channels,
                    out_features=module.out_channels,
                    per_dim_list=self.config.per_dim_list,
                    model_mode=self.config.model_mode,
                    rank=self.config.rank,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                )
                parent.add_module("lora", lora_layer)

        # Freeze original weights
        for n, p in self.model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False

    def forward(self, x):
        for name, module in self.model.named_modules():
            if "lora" in name:
                parent_name = ".".join(name.split(".")[:-1])
                parent_module = self.model.get_submodule(parent_name)

                original_forward = parent_module.forward

                def new_forward(x_in):
                    return original_forward(x_in) + parent_module.lora(x_in)

                parent_module.forward = new_forward

        return self.model(x)


# ============== DEMONSTRATION SCRIPT ============== #

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate a standard 3D model (MONAI UNet)
    base_model = UNet(
        spatial_dims=3,
        in_channels=1,  # e.g., single-channel MRI
        out_channels=2,  # e.g., segmentation mask with 2 classes
        channels=(16, 32, 64, 128),  # Channel depth at each level
        strides=(2, 2, 2),  # Strides for downsampling
    ).to(device)

    # 2. Define which layers to apply TenVOO to
    # We target the 3D convolutional layers within the UNet's blocks
    # MONAI UNet layer names look like `model.0.conv.conv`, `model.2.conv.conv` etc.
    target_modules = [".*.model.*.conv.conv"]

    # 3. Create TenVOO config and wrap the model
    peft_config = TenVOOConfig(
        d_in=3,
        d_out=3,
        target_modules=target_modules,
        model_mode="l",  # TenVOO-L
        rank=4,  # LoRA rank
    )

    # Print parameters before applying PEFT
    total_params_before = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )
    print(f"\n--- Before TenVOO ---")
    print(f"Total trainable parameters: {total_params_before:,}")

    # Apply TenVOO
    peft_model = TenVOOModel(peft_config, base_model).to(device)

    # Print parameters after applying PEFT
    trainable_params_after = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    total_params_after = sum(p.numel() for p in peft_model.parameters())
    print(f"\n--- After TenVOO ---")
    print(f"Total trainable parameters: {trainable_params_after:,}")
    print(f"Total model parameters:     {total_params_after:,}")
    print(
        f"Parameter reduction ratio (trainable): {(1 - trainable_params_after / total_params_before) * 100:.2f}%"
    )

    # 4. Mock training loop
    print("\n--- Mock Training Step ---")
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=1e-4)
    dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)  # (B, C, D, H, W)
    dummy_target = torch.randint(0, 2, (1, 64, 64, 64)).to(device)  # (B, D, H, W)

    peft_model.train()
    optimizer.zero_grad()
    output = peft_model(dummy_input)
    # For UNet, output is (B, C, D, H, W), target is (B, D, H, W)
    # Use CrossEntropyLoss for segmentation
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, dummy_target)
    loss.backward()
    optimizer.step()
    print(f"Mock training step completed. Loss: {loss.item():.4f}")

    # 5. Merge weights back for inference
    print("\n--- Merging Weights for Inference ---")
    peft_model.eval()
    merged_model = peft2nnmodel(
        peft_model.model
    )  # Pass the original model from the wrapper
    print("TenVOO weights merged successfully into the base model.")

    # Verify inference works on the merged model
    with torch.no_grad():
        inference_output = merged_model(dummy_input)
    print(
        f"Inference on merged model completed. Output shape: {inference_output.shape}"
    )

    print("\n✅ TenVOO PoC Experiment Successful!")
