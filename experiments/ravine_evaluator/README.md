# Ravine Audio-Visual Evaluator

## Overview

The Ravine evaluator is an experimental framework designed to assess the performance of Vision Language Models (VLMs) on tasks requiring joint audio and visual reasoning. It is inspired by the capabilities of multi-modal foundation models like `FISHER`, which excel at interpreting industrial audio signals.

This script serves as a proof-of-concept for extending `VQASynth`'s evaluation capabilities. It tests whether a VLM can correctly answer a question by synthesizing context from both an image and a descriptive text derived from an audio clip.

## Motivation

While `VQASynth` is powerful for generating spatial reasoning data from static images, real-world environments are multi-modal. An autonomous agent in a warehouse, for example, must react to both what it sees (a forklift) and what it hears (a backup alarm).

This evaluator establishes a baseline for this new, combined reasoning capability and provides a methodology for testing future VLMs trained on multi-modal datasets.

## Usage

1.  **Setup Environment:**
    Ensure you have Python, PyTorch with CUDA, and the necessary libraries installed.
    ```bash
    pip install torch torchaudio transformers requests Pillow
    ```

2.  **Run Evaluation:**
    Execute the script from within the `experiments/ravine_evaluator` directory. You can specify the Hugging Face model you wish to evaluate.

    ```bash
    python process_evaluation.py --model_id remyxai/SpaceThinker-Qwen2.5VL-3B
    ```

3.  **Check Results:**
    The script will print the evaluation progress and a final accuracy score to the console. Detailed results will be saved in `ravine_results.json`.
