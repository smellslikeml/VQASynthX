# Yume Dynamic Scene Generation Evaluation

This directory contains a script to evaluate the Yume image-to-video model for generating dynamic, navigable scenes from a static image. This serves as a proof-of-concept for creating simulation environments for embodied agents.

## Purpose

The goal is to take a static image, like one from the VQASynth dataset, and use the Yume model to generate a short video clip that simulates a basic agent action, such as moving forward.

## Setup

This experiment requires a Python environment with PyTorch, CUDA, and the `diffusers` library. A GPU with at least 16GB of VRAM is recommended.

1.  **Install dependencies:**

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install diffusers transformers accelerate imageio imageio-ffmpeg
    ```

2. **Authenticate with Hugging Face (if needed):**

   You may need to log in to download the model.
   ```bash
   huggingface-cli login
   ```

## Usage

Run the `process_evaluation.py` script, providing a path to an input image and a text prompt that describes the desired camera/agent motion.

### Example

Use one of the sample images from the VQASynth repository and a prompt formatted similarly to those in the Yume project.

```bash
python experiments/ravine_evaluator/process_evaluation.py \
  --image_path ../../assets/warehouse_sample_1.jpeg \
  --prompt "This video depicts a warehouse scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:3.0.Angular change rate (turn speed):0.0.View rotation speed:0.0." \
  --output_path ./output/warehouse_forward.mp4
```

After running, the generated video `warehouse_forward.mp4` will be saved in the `output` directory within the experiment folder. Visually inspect the video to see if it successfully animated the static image according to the prompt.