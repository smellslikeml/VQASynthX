# Ravine Evaluator

This directory contains an experiment to evaluate whether providing a Vision-Language Model (VLM) with a supplementary 2D map of object locations improves its spatial reasoning accuracy.

This is inspired by research showing that visualizations help AI models with quantitative data analysis. We are testing if the same principle applies to spatial VQA.

## Usage

### 1. Installation

Ensure you have the necessary dependencies installed. You may need to add `matplotlib`, `datasets`, and `pandas` to your environment.

```bash
pip install matplotlib datasets pandas
```

You will also need to be authenticated with Hugging Face to download the model and dataset.

```bash
huggingface-cli login
```

### 2. Run the Evaluation

Execute the script from the root of the repository:

```bash
python experiments/ravine_evaluator/process_evaluation.py
```

The script will:
1. Load the `remyxai/SpaceThinker-Qwen2.5VL-3B` model and the `remyxai/SpaceOm` dataset.
2. For a small sample of the data, it runs two inferences:
    - **Baseline:** The model answers a question using only the photo.
    - **Experiment:** The model answers the same question but is also given a programmatically generated 2D top-down map of the objects in the scene.
3. It will then calculate the Mean Absolute Error (MAE) for distance estimation for both conditions and print a comparison.

### 3. Expected Output

You should see a summary printed to the console comparing the performance of the two approaches.

```
--- Evaluation Complete ---
Processed 18 valid samples.
MAE (Baseline - Image Only): 1.2345
MAE (Experiment - Image + 2D Map): 0.8765

Improvement (lower MAE is better): 0.3580 (29.00%)

Sample Results:
... (a pandas DataFrame will be printed) ...
```
