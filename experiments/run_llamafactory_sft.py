import argparse
import os
import subprocess
import sys


def run_sft(args):
    """
    Constructs and runs the LLaMA-Factory SFT command based on provided arguments.
    """
    # Base command for LLaMA-Factory CLI, configured for VLM SFT with LoRA.
    command = [
        "llamafactory-cli",
        "train",
        "--stage",
        "sft",
        "--do_train",
        "--model_name_or_path",
        args.model_name_or_path,
        "--dataset",
        args.dataset,
        "--dataset_dir",
        args.dataset_dir,
        "--template",
        args.template,
        "--finetuning_type",
        "lora",
        "--lora_target",
        "all",
        "--lora_rank",
        "64",
        "--lora_alpha",
        "128",
        "--output_dir",
        args.output_dir,
        "--overwrite_output_dir",
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_acc_steps),
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "10",
        "--save_steps",
        "100",
        "--learning_rate",
        "5e-5",
        "--num_train_epochs",
        "3.0",
        "--plot_loss",
        "--fp16",
        "--ddp_find_unused_parameters",
        "False",
    ]

    # Add optional flash attention 2 for efficiency if supported
    if args.flash_attn:
        command.append("--flash_attn")

    # Add a reporting tool if specified
    if args.report_to:
        command.extend(["--report_to", args.report_to])

    print("--- " * 10)
    print("Running LLaMA-Factory SFT with the following command:")
    try:
        import shlex

        print(" ".join(shlex.quote(c) for c in command))
    except ImportError:
        print(" ".join(command))
    print("--- " * 10)

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("--- " * 10)
        print(f"SFT completed. LoRA adapter saved to {args.output_dir}")
        print("--- " * 10)
    except subprocess.CalledProcessError as e:
        print(f"Error during LLaMA-Factory training: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'llamafactory-cli' not found.", file=sys.stderr)
        print("Please ensure LLaMA-Factory is installed. For example:", file=sys.stderr)
        print("pip install 'llamafactory[vllm]'", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run Supervised Fine-Tuning (SFT) using LLaMA-Factory on VQASynth data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model ID or path for the VLM to be fine-tuned.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset, as defined in dataset_info.json (e.g., 'vqasynth_spatial').",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/",
        help="Directory containing the training data and the dataset_info.json file.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="llava",
        help="Prompt template to use. Should match the model type (e.g., 'llava', 'qwen').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/llava-1.5-7b-sft-lora",
        help="Directory to save the LoRA adapter and training checkpoints.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Per-device training batch size."
    )
    parser.add_argument(
        "--grad_acc_steps", type=int, default=8, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Enable Flash Attention 2 for faster training.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Experiment tracker to use (e.g., 'wandb', 'tensorboard', 'none').",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)

    dataset_info_path = os.path.join(args.dataset_dir, "dataset_info.json")
    print(f"Checking for dataset configuration at: {dataset_info_path}")
    if not os.path.exists(dataset_info_path):
        print(f"\nWARNING: '{dataset_info_path}' not found.", file=sys.stderr)
        print(
            "Please create it to map your dataset name to a file. For example:",
            file=sys.stderr,
        )
        print(
            '{\n  "'
            + args.dataset
            + '": {\n    "file_name": "your_training_data.jsonl"\n  }\n}',
            file=sys.stderr,
        )
        print(
            "\nPlace this file and 'your_training_data.jsonl' in the '--dataset_dir'.",
            file=sys.stderr,
        )
        print(
            "The data file should be a .jsonl where each line is a JSON object like:",
            file=sys.stderr,
        )
        print(
            '{"image": "path/to/img.jpg", "conversations": [{"from": "human", "value": "<image>\\nQuestion?"}, {"from": "gpt", "value": "Answer."}]}',
            file=sys.stderr,
        )
        sys.exit(1)

    run_sft(args)


if __name__ == "__main__":
    main()
