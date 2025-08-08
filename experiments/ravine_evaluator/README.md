# Ravine Evaluator

This directory contains a minimal script to evaluate the performance of VQASynth-trained models on spatial reasoning datasets.

The script loads a specified model and dataset from the Hugging Face Hub, runs inference on a number of samples, and computes metrics to compare the model's answers against the ground truth.

## Rationale

The `VQASynth` project excels at generating high-quality spatial VQA datasets. However, to iterate effectively, a standardized evaluation framework is needed to benchmark the models trained on this data. This "Ravine Evaluator" is a first step towards that goal, inspired by comparative analysis methods. It provides a way to quantitatively measure model improvements and regressions.

## Usage

### 1. Setup

Ensure you have the main project dependencies installed. You will also need `accelerate` for faster inference on GPUs.

```bash
# From the root of the VQASynth repo
pip install -r requirements.txt
pip install accelerate pandas
```

### 2. Run Evaluation

Execute the `process_evaluation.py` script from the command line. You can specify the model, dataset, and number of samples to evaluate.

```bash
# Example evaluation
python experiments/ravine_evaluator/process_evaluation.py \
    --model_id "remyxai/SpaceThinker-Qwen2.5VL-3B" \
    --dataset_id "remyxai/OpenSpaces_MC_R1" \
    --num_samples 50
```

### 3. Check Outputs

The script will print a summary of the evaluation metrics to the console, including:
- **Mean Absolute Error (MAE)** for quantitative questions (e.g., distance estimation).
- **Accuracy** for qualitative questions (e.g., yes/no, relational comparisons).

A detailed log of each question, ground truth, model answer, and calculated error will be saved to `evaluation_results.csv` in the directory from which the script was run.