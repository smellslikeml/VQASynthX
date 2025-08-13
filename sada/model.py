# This is a simplified, representative implementation based on the source repo's structure.
import torch
import torch.nn as nn
from . import module


class SADAUNet(nn.Module):
    def __init__(self, unet, cache_bus):
        super().__init__()
        self.unet = unet
        self.cache_bus = cache_bus

        # Recursively patch attention blocks
        for name, sub_module in self.unet.named_modules():
            if "Attention" in sub_module.__class__.__name__:
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent_module = self.unet.get_submodule(parent_name)

                patched_attn = module.SADAAttention(sub_module, self.cache_bus)
                setattr(parent_module, child_name, patched_attn)

    def forward(self, *args, **kwargs):
        self.cache_bus.step()  # Increment step counter
        return self.unet(*args, **kwargs)
