import os
import json
import sys

# Attempt to import LLaMA-Factory
try:
    from llamafactory.train.tuner import run_exp
except ImportError:
    print("Error: LLaMA-Factory is not installed.", file=sys.stderr)
    print(
        "Please install it, e.g., pip install 'llamafactory[torch,bitsandbytes]'",
        file=sys.stderr,
    )
    sys.exit(1)


def setup_dataset_info(dataset_file_path: str, dataset_name: str) -> str:
    """
    LLaMA-Factory requires a dataset_info.json file in the dataset directory
    to map a dataset name to a file. This function creates it dynamically.
    The dataset is expected to be in ShareGPT format with an "image" key.
    """
    dataset_dir = os.path.dirname(dataset_file_path)
    file_name = os.path.basename(dataset_file_path)

    # This configuration tells LLaMA-Factory how to parse the VLM dataset.
    # It assumes the VQASynth output is a JSON list of conversations.
    # Each conversation contains turns from "human" and "gpt".
    # The first human turn should contain an <image> placeholder.
    dataset_info = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",  # Use ShareGPT format for image conversations
            "columns": {"messages": "conversations"},
        }
    }

    info_file_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(info_file_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Generated dataset_info.json for LLaMA-Factory at: {info_file_path}")
    return dataset_dir


def main():
    """
    Main function to configure and run the SFT job.
    Modify the `TRAINING_ARGS` dictionary below to configure the training run.
    """
    # --- START OF USER CONFIGURATION ---

    # 1. Path to the VQASynth-generated dataset.
    #    The dataset JSON file should be in ShareGPT format, e.g.:
    #    [{"conversations": [{"from": "human", "value": "<image>\nWhat is in the image?"}, {"from": "gpt", "value": "..."}]}]
    #    The path to the image file inside the JSON must be relative to the location of this script or absolute.
    dataset_file = "data/sample_spatial_vqa.json"

    # 2. Define training arguments for LLaMA-Factory.
    #    See LLaMA-Factory documentation for all possible options.
    training_args = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": "llava-hf/llava-1.5-7b-hf",
        "output_dir": "models/vqasynth-llava-1.5-7b-lora",
        "finetuning_type": "lora",
        "lora_target": "all",
        "template": "llava",
        "visual_inputs": True,  # This is crucial for VLM training
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 1.0,
        "fp16": True,
        "logging_steps": 10,
        "save_steps": 100,
        "plot_loss": True,
    }
    # --- END OF USER CONFIGURATION ---

    # --- SCRIPT LOGIC ---
    print("--- VQASynth SFT Training with LLaMA-Factory ---")

    # Verify dataset file exists
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at '{dataset_file}'", file=sys.stderr)
        print(
            "Please generate the dataset using the VQASynth pipeline or update the 'dataset_file' path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Prepare LLaMA-Factory's dataset_info.json
    dataset_name = "vqasynth_sft_dataset"
    abs_dataset_path = os.path.abspath(dataset_file)
    dataset_dir = setup_dataset_info(abs_dataset_path, dataset_name)

    # Add dataset-specific args to the main config
    training_args["dataset"] = dataset_name
    training_args["dataset_dir"] = dataset_dir

    # Run the experiment using LLaMA-Factory's programmatic API
    print("\nStarting LLaMA-Factory fine-tuning job with the following config:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")

    run_exp(training_args)

    print("\n--- Fine-tuning complete! ---")
    print(f"LoRA adapter and training logs saved to: {training_args['output_dir']}")


if __name__ == "__main__":
    main()
