# LLaMA-Factory SFT Experiment

This directory contains a self-contained experiment for fine-tuning a Vision-Language Model (VLM) using LLaMA-Factory on a VQASynth-generated dataset.

## Objective

To demonstrate a complete pipeline from VQASynth's spatial data generation to model fine-tuning by integrating the powerful `LLaMA-Factory` framework.

## How to Run

1.  **Build the Docker Image:**
    From this directory (`experiments/llama_factory_sft`), run:
    ```bash
    docker build -t vqasynth-sft-experiment .
    ```

2.  **Run the Training:**
    Execute the training script within the container. This command mounts a local `saves` directory to persist the trained model adapters. Ensure you have a `saves` directory in the current folder.
    ```bash
    mkdir -p saves
    docker run --rm --gpus all -v $(pwd)/saves:/app/saves vqasynth-sft-experiment
    ```

3.  **Check Results:**
    Once the run completes, the trained LoRA adapters will be available in `./saves/Qwen-VL-Chat/lora/sft/`.
