# RAVine-Inspired Evaluator

This script provides a framework for quantitatively evaluating the performance of models on VQASynth-generated datasets. It is inspired by the structured evaluation methodology of the [RAVine](https://github.com/SwordFaith/RAVine) project, adapting its principles for the VQA domain.

The script calculates exact match accuracy by comparing model-generated answers against the ground-truth answers in a specified Hugging Face dataset.

## Usage

### Prerequisites
Install necessary Python libraries:
```bash
pip install pandas datasets
```

### Running the Evaluation

To run the evaluation, you need two files:
1. A ground-truth dataset from the Hugging Face Hub (e.g., `remyxai/OpenSpaces_MC_R1`).
2. A predictions file in JSONL format. Each line should be a JSON object containing an `id` that matches the dataset and a `prediction` string from your model. The prediction should follow the `<think>...</think> <answer>...</answer>` format.

**Example `predictions.jsonl`:**
```json
{"id": "OPENSPACES_MC_R1_000001", "prediction": "<think>The user is asking about the color of the car.</think> <answer>The car is red.</answer>"}
{"id": "OPENSPACES_MC_R1_000002", "prediction": "<think>The user is asking a yes/no question.</think> <answer>Yes.</answer>"}
```

**Command:**
```bash
python experiments/ravine_evaluator/process_evaluation.py \
    --predictions_file path/to/your/predictions.jsonl \
    --dataset_name remyxai/OpenSpaces_MC_R1 \
    --dataset_split test \
    --output_file metrics.json
```

### Output

The script will generate a `metrics.json` file with the evaluation results, similar to this:

```json
{
    "total_evaluated": 100,
    "correct_predictions": 85,
    "exact_match_accuracy": "85.00%"
}
```
