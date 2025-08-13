# This script applies the OptSpa pruning method to a fine-tuned LLaVA model.
# It is designed to be run as an additional step after the VQASynth training
# pipeline to produce a compressed, efficient model for deployment.
#
# This script requires the OptSpa repository to be available in the python
# environment. Clone it and install its requirements first:
# > git clone https://github.com/tezhang65/optspa.git
# > # install dependencies from optspa/INSTALL.md
# > export PYTHONPATH=$PYTHONPATH:$(pwd)/optspa
#

import argparse
import os
import torch
import numpy as np

# These imports are from the OptSpa repository.
# They will be found if OptSpa is in the PYTHONPATH.
from lib.prune import prune_optspa, check_sparsity
from lib.tool import (
    get_llava,
)  # get_llava is a custom loader from optspa for LLaVA models


def run_pruning():
    """
    Main function to parse arguments and run the pruning process.
    """
    parser = argparse.ArgumentParser(
        description="Apply OptSpa pruning to a fine-tuned VQASynth LLaVA model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the fine-tuned VQASynth LLaVA model.",
    )
    parser.add_argument(
        "--sparsity_ratio",
        type=str,
        required=True,
        help="Path to the directory containing the fine-tuned VQASynth LLaVA model.",
    )
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.5,
        help="Target unstructured sparsity level for the model.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the pruned model checkpoint.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples for pruning. The optspa default is 128.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling calibration data."
    )
    # The OptSpa library requires the following arguments to be passed.
    parser.add_argument(
        "--sparsity_type",
        type=str,
        default="unstructured",
        choices=["unstructured", "4:8", "2:4"],
        help="Sparsity type, 'unstructured' is recommended for OptSpa.",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        default="optspa",
        help="Pruning method, fixed to 'optspa' for this script.",
    )

    args = parser.parse_args()

    # Set seeds for reproducibility, as done in the source repository.
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemError("CUDA is required for this pruning operation.")
    device = "cuda"

    print(f"Loading LLaVA model from: {args.model_path}")
    # Use the custom loader from the OptSpa repository to correctly handle LLaVA models.
    # This assumes the model is a LLaVA-v1.5 variant, which is consistent with both
    # the source (OptSpa) and target (VQASynth) repositories.
    model, tokenizer = get_llava(args.model_path)
    model.to(device)
    model.eval()

    print("\nModel loaded successfully. Starting OptSpa pruning process...")

    # The prune_optspa function from the source repo modifies the model in-place
    # and returns the pruned model. It uses Optuna to find the best layer-wise
    # sparsity distribution.
    pruned_model = prune_optspa(args, model, tokenizer, device, prune_n=0, prune_m=0)

    print("\nPruning complete. Checking final sparsity...")
    final_sparsity = check_sparsity(pruned_model)
    print(f"Sparsity sanity check: {final_sparsity:.4f}")

    if args.save_path:
        print(f"\nSaving pruned model to: {args.save_path}")
        os.makedirs(args.save_path, exist_ok=True)
        pruned_model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"Pruned model and tokenizer saved successfully to {args.save_path}")


if __name__ == "__main__":
    run_pruning()
