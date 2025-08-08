#!/usr/bin/env python3
"""
Generate multiple config files with different hyperparameter combinations
for the VQASynth data generation pipeline.

Inspired by the systematic experimentation approach in the MedARC Algonauts 2025 repository.
"""

import yaml
import os
import itertools
from pathlib import Path

ROOT = Path(__file__).parent
# Assume a base config file exists for the pipeline
BASE_CONFIG_PATH = ROOT / "config/base_pipeline.yaml"
OUTPUT_DIR = ROOT / "configs/generated/"


def load_base_config(config_path):
    """Load the base configuration file."""
    if not config_path.exists():
        print(f"Warning: Base config not found at {config_path}. Using an empty dict.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_config_variations():
    """Define all the pipeline variations to test."""

    variations = {
        "depth_model": ["VGGT", "DepthAnything", "DepthPro"],
        "localization_model": ["SAM2", "SAM"],
        "min_objects_per_image": [2, 3],
        "prompt_template_version": ["v1", "v2-cot"],
        "max_questions_per_image": [5, 10]
    }

    # Generate all combinations of the defined variations
    keys, values = zip(*variations.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return param_combinations


def create_config_name(base_name, params):
    """Create a descriptive config name from the parameters."""
    param_str = "_".join([f"{k}-{v}" for k, v in params.items()])
    return f"{base_name}_{param_str}.yaml"


def modify_config(base_config, params):
    """Modify the base config with a new set of parameters."""
    config = base_config.copy()

    # Update pipeline stage parameters based on the variations
    config["depth_stage"] = config.get("depth_stage", {})
    config["depth_stage"]["model"] = params["depth_model"]

    config["location_refinement_stage"] = config.get("location_refinement_stage", {})
    config["location_refinement_stage"]["model"] = params["localization_model"]

    config["filter_stage"] = config.get("filter_stage", {})
    config["filter_stage"]["min_objects"] = params["min_objects_per_image"]
    
    config["prompt_stage"] = config.get("prompt_stage", {})
    config["prompt_stage"]["template_version"] = params["prompt_template_version"]
    config["prompt_stage"]["max_questions"] = params["max_questions_per_image"]

    # Update an output path to keep experiment results separate
    param_suffix = "_".join([f"{k}-{v}" for k, v in params.items()])
    config["output_dataset_name"] = f"vqasynth_dataset_{param_suffix}"

    return config


def main():
    """Generate and save all configuration files."""
    if not BASE_CONFIG_PATH.exists():
        # Create a dummy base config if it doesn't exist to make the script runnable
        print(f"Creating dummy base config at {BASE_CONFIG_PATH}")
        os.makedirs(BASE_CONFIG_PATH.parent, exist_ok=True)
        dummy_config = {
            "dataset_source": "huggingface/some_image_dataset",
            "depth_stage": {"model": "VGGT"},
            "location_refinement_stage": {"model": "SAM2"},
            "filter_stage": {"min_objects": 2},
            "prompt_stage": {"template_version": "v1", "max_questions": 5},
            "output_dataset_name": "vqasynth_dataset_base"
        }
        with open(BASE_CONFIG_PATH, 'w') as f:
            yaml.dump(dummy_config, f, default_flow_style=False)

    base_config = load_base_config(BASE_CONFIG_PATH)
    param_combinations = generate_config_variations()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating {len(param_combinations)} config files in {OUTPUT_DIR}...")

    for params in param_combinations:
        modified_conf = modify_config(base_config, params)
        config_name = create_config_name("pipeline_config", params)
        output_path = OUTPUT_DIR / config_name

        with open(output_path, "w") as f:
            yaml.dump(modified_conf, f, default_flow_style=False, sort_keys=False)

    print("Done.")


if __name__ == "__main__":
    main()
