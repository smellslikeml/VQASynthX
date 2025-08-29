import os
import subprocess
import requests
from pathlib import Path

# --- Configuration ---
# URL of a sample image from the VQASynth repository assets
IMAGE_URL = "https://raw.githubusercontent.com/smellslikeml/experimental-vqasynth/main/assets/warehouse_sample_1.jpeg"
IMAGE_NAME = "warehouse_sample_1.jpeg"

# Directories for the experiment, relative to the container's /app workdir
EXPERIMENT_DIR = Path("experiments/yume_sim")
INPUT_DIR = EXPERIMENT_DIR / "input"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
YUME_REPO_DIR = "yume_repo"  # Name of the cloned YUME directory


# --- Main Script ---
def setup_directories():
    """Create necessary directories for input and output."""
    print("Setting up directories...")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_sample_image():
    """Download a sample image to be used for video generation."""
    image_path = INPUT_DIR / IMAGE_NAME
    if not image_path.exists():
        print(f"Downloading sample image from {IMAGE_URL}...")
        try:
            response = requests.get(IMAGE_URL, timeout=30)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"Image saved to {image_path}")
        except requests.RequestException as e:
            print(f"Error downloading image: {e}")
            exit(1)
    else:
        print(f"Image {image_path} already exists.")
    return image_path


def create_action_caption():
    """Create the caption file with a specific action prompt."""
    caption_content = (
        "This video depicts a warehouse scene with a first-person view (FPV)."
        "Person moves forward (W).Camera remains still (·)."
        "Actual distance moved:2.0.Angular change rate (turn speed):0.5."
        "View rotation speed:0.5."
    )
    caption_path = INPUT_DIR / "caption.txt"
    with open(caption_path, "w") as f:
        f.write(caption_content)
    print(f"Action caption written to {caption_path}")
    return caption_path


def run_yume_inference(image_dir, caption_path, output_dir):
    """
    Run the YUME image-to-video inference script using torchrun.
    This function assumes the YUME repository is at YUME_REPO_DIR
    and dependencies are installed, as handled by the Dockerfile.
    """
    print("\nStarting YUME inference process...")

    # Use absolute paths within the container for clarity
    abs_image_dir = Path.cwd() / image_dir
    abs_caption_path = Path.cwd() / caption_path
    abs_output_dir = Path.cwd() / output_dir

    command = [
        "torchrun",
        "--nproc_per_node=1",
        "fastvideo/sample/sample.py",
        f"--video_output_dir={abs_output_dir}",
        f"--jpg_dir={abs_image_dir}",
        f"--caption_path={abs_caption_path}",
        "--num_euler_timesteps",
        "50",
        "--seed",
        "42",
        "--mixed_precision=bf16",
        "--allow_tf32",
        "--t5_cpu",
    ]

    print(f"Executing command: {' '.join(command)}")
    print(f"Working directory: {YUME_REPO_DIR}")

    try:
        # The inference script can take a few minutes, especially on the first run downloading models.
        process = subprocess.run(
            command,
            cwd=YUME_REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minutes timeout
        )
        print("--- YUME stdout ---")
        print(process.stdout)
        print("--- YUME stderr ---")
        print(process.stderr)
        print("\n✅ YUME inference completed successfully.")
        print(f"🎥 Video output saved in: {abs_output_dir}")

    except FileNotFoundError:
        print("❌ Error: `torchrun` not found. Is the environment set up correctly?")
        exit(1)
    except subprocess.CalledProcessError as e:
        print("❌ Error: YUME inference script failed.")
        print("--- stdout ---")
        print(e.stdout)
        print("--- stderr ---")
        print(e.stderr)
        exit(1)
    except subprocess.TimeoutExpired as e:
        print("❌ Error: YUME inference timed out.")
        print("--- stdout ---")
        print(e.stdout)
        print("--- stderr ---")
        print(e.stderr)
        exit(1)


if __name__ == "__main__":
    setup_directories()
    download_sample_image()
    caption_path = create_action_caption()
    run_yume_inference(
        image_dir=INPUT_DIR, caption_path=caption_path, output_dir=OUTPUT_DIR
    )
