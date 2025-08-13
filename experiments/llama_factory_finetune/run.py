import os
import json
import subprocess
import sys

try:
    from PIL import Image
except ImportError:
    print(
        "Warning: Pillow library not found. Run 'pip install Pillow' to create dummy image file."
    )
    Image = None


def setup_experiment():
    """
    Creates a self-contained test environment for LLaMA-Factory finetuning.
    This includes a dummy dataset, a dummy image, and the required config file.
    """
    # Use relative paths to ensure the experiment is self-contained
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(exp_dir, "data")
    output_dir = os.path.join(exp_dir, "output")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create a dummy image file
    image_path = os.path.join(data_dir, "dummy_image.png")
    if not os.path.exists(image_path):
        if Image:
            img = Image.new("RGB", (20, 20), color="gray")
            img.save(image_path)
            print(f"Created dummy image at: {image_path}")
        else:
            print("Cannot create dummy image because Pillow is not installed.")
            sys.exit(1)

    # 2. Create a dummy dataset file (vqasynth_data.jsonl)
    # This format is compatible with LLaMA-Factory's 'llava' template.
    dataset_path = os.path.join(data_dir, "vqasynth_data.jsonl")
    sample_data = {
        "image": "dummy_image.png",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nBased on the image, describe the spatial relationship between the objects.",
            },
            {
                "from": "gpt",
                "value": "This is a placeholder response for a spatial reasoning task. The objects are arranged in a test pattern.",
            },
        ],
    }
    with open(dataset_path, "w") as f:
        f.write(json.dumps(sample_data) + "\n")
    print(f"Created dummy dataset at: {dataset_path}")

    # 3. Create the dataset_info.json configuration file for LLaMA-Factory
    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    dataset_info = {
        "vqasynth_spatial": {
            "file_name": "vqasynth_data.jsonl",
            "columns": {"image": "image", "messages": "conversations"},
        }
    }
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Created dataset info at: {dataset_info_path}")

    return data_dir, output_dir


def run_finetuning(data_dir, output_dir):
    """
    Constructs and runs the llamafactory-cli command for a test finetuning job.
    """
    # These arguments are chosen for a minimal, fast-to-run test.
    args = [
        "llamafactory-cli",
        "train",
        "--model_name_or_path",
        "llava-hf/llava-1.5-7b-hf",
        "--do_train",
        "True",
        "--dataset_dir",
        data_dir,
        "--dataset",
        "vqasynth_spatial",
        "--template",
        "llava",
        "--finetuning_type",
        "lora",
        "--lora_target",
        "all",
        "--output_dir",
        output_dir,
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--logging_steps",
        "1",
        "--max_steps",
        "3",
        "--save_steps",
        "2",
        "--learning_rate",
        "1e-4",
        "--fp16",
    ]

    print("\n--- Running LLaMA-Factory Finetuning ---")
    print(f"Command: {' '.join(args)}")

    try:
        # Execute the command from the experiment directory context
        subprocess.run(args, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print("\n--- Finetuning Completed Successfully ---")
        print(f"Check for adapter model in: {output_dir}")
    except FileNotFoundError:
        print("\nError: `llamafactory-cli` not found.")
        print(
            "Please ensure LLaMA-Factory is installed correctly in your environment's PATH."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n--- Finetuning Failed ---")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    data_directory, output_directory = setup_experiment()
    run_finetuning(data_directory, output_directory)
