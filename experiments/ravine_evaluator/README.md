# Ravine Evaluator

This directory contains an experiment to generate synthetic evaluation data for VQASynth-trained models. The goal is to create a fast, reproducible way to generate images with known spatial ground truth, which can be used to benchmark the spatial reasoning capabilities of Vision-Language Models.

This experiment integrates `SADA` (Stability-guided Adaptive Diffusion Acceleration) to accelerate the image generation process using the `FLUX.1-dev` diffusion model from Hugging Face.

## How it Works

The `process_evaluation.py` script performs the following steps:
1.  Generates an image using the standard `diffusers` pipeline for a baseline measurement.
2.  Applies the `SADA` patch to a new pipeline instance.
3.  Generates the same image using the SADA-accelerated pipeline.
4.  Compares the generation time, calculates the speedup, and measures the visual similarity using the LPIPS metric.
5.  Saves a side-by-side comparison image `ravine_comparison.png`.

## Usage

### 1. Installation

Ensure you have a Python environment with PyTorch and CUDA support. Then, install the required packages:

```bash
# Install core dependencies for the script
pip install diffusers transformers accelerate torch torchvision lpips

# Install SADA directly from the source repository
pip install git+https://github.com/Ting-Justin-Jiang/sada-icml.git
```

### 2. Run the Experiment

Execute the script from the repository root:

```bash
python experiments/ravine_evaluator/process_evaluation.py
```

### 3. Check the Output

The script will print the following to the console:
-   Baseline generation time.
-   SADA-accelerated generation time.
-   The calculated speedup factor (e.g., `Speedup: 1.85x`).
-   The LPIPS distance, where a lower score indicates higher similarity to the baseline.

Additionally, an image file named `ravine_comparison.png` will be created in the current directory, allowing for a visual comparison between the baseline (left) and SADA-accelerated (right) images.