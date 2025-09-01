#!/usr/bin/env python
# Inspired by the functional network pruning concept from https://github.com/WhatAboutMyStar/LLM_ACTIVATION
# This script implements a simplified version using activation correlation to identify and prune neurons.

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import os
import gc

# --- Configuration ---
MODEL_ID = "lmsys/vicuna-7b-v1.5"
CALIBRATION_DATASET = "Self-GRIT/wikitext-2-raw-v1-preprocessed"
NUM_SAMPLES = 50
MAX_LENGTH = 256
PRUNE_RATIO = 0.25  # Prune 25% of neurons with the lowest connectivity
CORRELATION_THRESHOLD = 0.8  # Threshold to consider two neurons 'connected'
OUTPUT_DIR = "/app/pruned_model"

# --- Main Pruning Logic ---


class ActivationCapturer:
    """A helper class to capture activations using forward hooks."""

    def __init__(self, model, target_module_pattern):
        self.model = model
        self.target_module_pattern = target_module_pattern
        self.activations = {}
        self.hooks = []

    def _get_hook(self, name):
        def hook(module, input, output):
            # Capture the first element of the output tuple if it's a tuple
            act = output[0] if isinstance(output, tuple) else output
            # Store activations on CPU to conserve VRAM
            self.activations[name] = act.detach().view(-1, act.shape[-1]).cpu()

        return hook

    def register_hooks(self):
        print(
            f"Registering hooks for modules matching '{self.target_module_pattern}'..."
        )
        for name, module in self.model.named_modules():
            if self.target_module_pattern in name:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))
        print(f"Registered {len(self.hooks)} hooks.")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


def get_calibration_data(dataset_name, num_samples, tokenizer, max_length):
    """Load and prepare calibration data."""
    print(f"Loading calibration data from '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train")
    samples = [dataset[i]["text"] for i in range(num_samples)]
    inputs = tokenizer(
        samples,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    return inputs


def prune_model_with_correlation(model, tokenizer, config):
    """Main function to perform correlation analysis and prune the model."""
    # 1. Setup activation capturing
    # In Llama models, the MLP has gate_proj, up_proj, and down_proj.
    # We capture activations from the gate_proj, which precedes the SiLU activation function.
    capturer = ActivationCapturer(model, target_module_pattern="mlp.gate_proj")
    capturer.register_hooks()

    # 2. Get calibration data
    calib_inputs = get_calibration_data(
        config["calibration_dataset"],
        config["num_samples"],
        tokenizer,
        config["max_length"],
    )

    # 3. Run calibration forward passes
    print("Collecting activations...")
    all_activations = {name: [] for name in capturer.activations.keys()}
    for i in tqdm(range(config["num_samples"]), desc="Calibration Pass"):
        input_ids_sample = calib_inputs.input_ids[i : i + 1].to(model.device)
        with torch.no_grad():
            model(input_ids_sample)
        for name, act in capturer.activations.items():
            all_activations[name].append(act)
    capturer.remove_hooks()

    # 4. Analyze activations and create pruning masks
    pruning_masks = {}
    for name, activations_list in tqdm(
        all_activations.items(), desc="Analyzing Layers"
    ):
        if not activations_list:
            continue
        # Concatenate activations from all samples for the current layer
        layer_activations = torch.cat(activations_list, dim=0).float()

        # Compute Pearson correlation matrix
        corr_matrix = torch.from_numpy(np.corrcoef(layer_activations.T.numpy()))
        corr_matrix.fill_diagonal_(0)  # Ignore self-correlation

        # Calculate a 'connectivity score' for each neuron
        connectivity_score = (corr_matrix.abs() > config["corr_threshold"]).sum(dim=1)

        # Identify neurons to prune (those with the lowest scores)
        num_neurons = connectivity_score.shape[0]
        num_to_prune = int(num_neurons * config["prune_ratio"])
        if num_to_prune == 0:
            continue

        neurons_to_prune = torch.argsort(connectivity_score)[:num_to_prune]

        # Create a boolean mask (True = keep, False = prune)
        mask = torch.ones(num_neurons, dtype=torch.bool)
        mask[neurons_to_prune] = False
        pruning_masks[name] = mask
        print(f"Layer {name}: Pruning {num_to_prune}/{num_neurons} neurons.")

    # 5. Apply pruning masks by zeroing weights
    print("Applying pruning masks to model weights...")
    state_dict = model.state_dict()
    for name, mask in tqdm(pruning_masks.items(), desc="Pruning Weights"):
        # Corresponding weights are in gate_proj, up_proj, and down_proj
        gate_proj_name = name
        up_proj_name = name.replace("gate_proj", "up_proj")
        down_proj_name = name.replace("gate_proj", "down_proj")

        # Prune columns for gate_proj and up_proj
        state_dict[gate_proj_name][:, ~mask] = 0
        state_dict[up_proj_name][:, ~mask] = 0
        # Prune rows for down_proj
        state_dict[down_proj_name][~mask, :] = 0

    model.load_state_dict(state_dict)
    print("Pruning complete.")


def main():
    """Entrypoint of the script."""
    print(f"Starting pruning process for model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    pruning_config = {
        "calibration_dataset": CALIBRATION_DATASET,
        "num_samples": NUM_SAMPLES,
        "max_length": MAX_LENGTH,
        "prune_ratio": PRUNE_RATIO,
        "corr_threshold": CORRELATION_THRESHOLD,
    }

    prune_model_with_correlation(model, tokenizer, pruning_config)

    # 6. Save the pruned model
    print(f"Saving pruned model and tokenizer to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Script finished successfully.")


if __name__ == "__main__":
    main()
