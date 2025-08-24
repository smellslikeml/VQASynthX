import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seeds for deterministic experiments, inspired by the robust
    experimental setup in the CNSDiff repository.

    Ensures reproducibility across Python's random module, NumPy, and PyTorch
    (for both CPU and GPU).

    Args:
        seed (int): The seed value to use.
    """
    print(f"Setting all random seeds to {seed} for reproducibility.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        # The following two lines are known to degrade performance but are
        # essential for true determinism in CUDA operations.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    """
    Demonstrates the effect of the set_seed utility. Running this script
    multiple times should produce the exact same "random" outputs. This is
    critical for debugging and comparing experiment runs in the VQASynth pipeline.
    """
    SEED = 2024

    print("--- Demonstrating Reproducibility ---")
    set_seed(SEED)

    # Generate a "random" numpy array
    numpy_random_array = np.random.rand(2, 3)
    print("\nGenerated NumPy array:")
    print(numpy_random_array)

    # Generate a "random" torch tensor
    torch_random_tensor = torch.rand(2, 3)
    print("\nGenerated PyTorch tensor:")
    print(torch_random_tensor)

    print("\n--- Rerunning with the same seed ---")
    set_seed(SEED)

    # Generate another "random" numpy array
    numpy_random_array_2 = np.random.rand(2, 3)
    print("\nGenerated NumPy array:")
    print(numpy_random_array_2)

    # Generate another "random" torch tensor
    torch_random_tensor_2 = torch.rand(2, 3)
    print("\nGenerated PyTorch tensor:")
    print(torch_random_tensor_2)

    # Verify that the outputs were identical
    assert np.array_equal(numpy_random_array, numpy_random_array_2)
    assert torch.equal(torch_random_tensor, torch_random_tensor_2)

    print("\nSuccess: Both runs produced identical outputs as expected.")
    print(
        "This utility can be integrated into VQASynth data generation and training scripts."
    )
