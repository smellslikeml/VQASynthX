import os
import subprocess
import yaml
import pandas as pd
from pathlib import Path
import tempfile
import json
from PIL import Image

# Configuration for the experiment
LTLZINC_REPO_PATH = Path("/app/LTLZinc")  # Assumed path in Docker environment
OUTPUT_DIR = Path("./data/ltlzinc_vqa")
NUM_SAMPLES = 10  # Generate a small number of samples for a quick test

# LTLZinc configuration for the MNIST addition alternation task
# Formula: "G (p(W,X,Y,Z) <-> WX !p(W,X,Y,Z))"
# This means the property p is true at time t if and only if it's false at time t+1.
# It enforces an alternating pattern, starting with p being true at t=0.
LTLZINC_CONFIG_TEMPLATE = """
mode: sequential
length: [5, 8] # Shorter sequences for easier visualization
splits:
  train: {{path: "{output_path}", samples: {num_samples}}}
minizinc_prefix: ""
predicates:
  "p(A,B,C,D)": "A + B = C + D"
formula: "G (p(W,X,Y,Z) <-> WX !p(W,X,Y,Z))"
types:
  mnist_t:
    0: "mnist/0"
    1: "mnist/1"
    2: "mnist/2"
    3: "mnist/3"
    4: "mnist/4"
    5: "mnist/5"
    6: "mnist/6"
    7: "mnist/7"
    8: "mnist/8"
    9: "mnist/9"
domains:
  W: mnist_t
  X: mnist_t
  Y: mnist_t
  Z: mnist_t
streams:
  W: w
  X: x
  Y: y
  Z: z
seed: 42
"""


def run_command(command, cwd):
    print(f"Running command: {' '.join(command)} in {cwd}")
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print("Error running command:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    return result


def generate_ltl_data(temp_dir):
    """
    Runs the LTLZinc generator to produce image sequences.
    """
    print("--- Step 1: Generating data with LTLZinc ---")

    if not LTLZINC_REPO_PATH.exists():
        raise FileNotFoundError(f"LTLZinc repository not found at {LTLZINC_REPO_PATH}.")

    print("Downloading MNIST images for LTLZinc...")
    run_command(["python", "download_images.py"], cwd=LTLZINC_REPO_PATH)

    ltlzinc_output_path = temp_dir / "ltlzinc_output"
    ltlzinc_output_path.mkdir()

    config_content = LTLZINC_CONFIG_TEMPLATE.format(
        output_path=str(ltlzinc_output_path), num_samples=NUM_SAMPLES
    )
    config_path = temp_dir / "config.yml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Generated LTLZinc config at {config_path}")

    # LTLZinc's main.py can take a config file as a command-line argument.
    run_command(["python", "main.py", str(config_path)], cwd=LTLZINC_REPO_PATH)

    print("LTLZinc data generation complete.")
    return ltlzinc_output_path


def create_composite_image(sequence_df, ltlzinc_data_path):
    """
    Creates a single composite image from a sequence of images.
    Each row in the returned image represents a timestep.
    """
    images_by_timestep = []
    for _, row in sequence_df.iterrows():
        w_path = ltlzinc_data_path / row["w"]
        x_path = ltlzinc_data_path / row["x"]
        y_path = ltlzinc_data_path / row["y"]
        z_path = ltlzinc_data_path / row["z"]

        img_w = Image.open(w_path).convert("RGB")
        img_x = Image.open(x_path).convert("RGB")
        img_y = Image.open(y_path).convert("RGB")
        img_z = Image.open(z_path).convert("RGB")

        width, height = img_w.size

        timestep_img = Image.new("RGB", (width * 2, height * 2))
        timestep_img.paste(img_w, (0, 0))
        timestep_img.paste(img_x, (width, 0))
        timestep_img.paste(img_y, (0, height))
        timestep_img.paste(img_z, (width, height))
        images_by_timestep.append(timestep_img)

    total_width = images_by_timestep[0].width
    total_height = sum(img.height for img in images_by_timestep)

    composite_image = Image.new("RGB", (total_width, total_height))
    current_y = 0
    for img in images_by_timestep:
        composite_image.paste(img, (0, current_y))
        current_y += img.height

    return composite_image


def create_vqa_dataset(ltlzinc_data_path):
    """
    Processes LTLZinc output into a VQA dataset.
    """
    print("--- Step 2: Creating VQA dataset from LTLZinc output ---")
    metadata_path = ltlzinc_data_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {ltlzinc_data_path}")

    metadata_df = pd.read_csv(metadata_path)

    vqa_image_dir = OUTPUT_DIR / "images"
    vqa_image_dir.mkdir(parents=True, exist_ok=True)

    vqa_data = []

    question = "In the image sequence (read top to bottom), each timestep shows four digits in a 2x2 grid (W, X, Y, Z). A rule states that the property 'W+X = Y+Z' must alternate between true and false at each step, starting as true. Is this rule followed correctly throughout the entire sequence?"
    answer = "Yes"

    for sequence_id in metadata_df["sequence_id"].unique():
        sequence_sub_df = metadata_df[metadata_df["sequence_id"] == sequence_id].copy()
        sequence_sub_df.sort_values("t", inplace=True)

        composite_img = create_composite_image(
            sequence_sub_df, ltlzinc_data_path.parent
        )
        image_filename = f"sequence_{sequence_id}.png"
        image_path = vqa_image_dir / image_filename
        composite_img.save(image_path)

        vqa_data.append(
            {
                "image": str(image_path.relative_to(OUTPUT_DIR.parent)),
                "question": question,
                "answer": answer,
                "sequence_id": int(sequence_id),
            }
        )

    output_jsonl_path = OUTPUT_DIR / "dataset.jsonl"
    with open(output_jsonl_path, "w") as f:
        for item in vqa_data:
            f.write(json.dumps(item) + "\n")

    print(f"VQA dataset created at {output_jsonl_path}")
    print(f"Composite images saved in {vqa_image_dir}")


def main():
    """Main execution function."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        ltlzinc_output_path = generate_ltl_data(temp_dir)
        create_vqa_dataset(ltlzinc_output_path)

    print("--- Experiment Complete ---")
    print(f"Find the generated VQA dataset in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
