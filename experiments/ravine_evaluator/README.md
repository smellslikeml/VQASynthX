# VQASynth Evaluation Framework

This experimental evaluation script is inspired by the structured, metric-driven evaluation methodology of the [RAVine](https://github.com/SwordFaith/RAVine) project. It provides a simple way to benchmark the performance of models trained with VQASynth, focusing on the key capability of quantitative spatial reasoning (i.e., distance estimation).

## Usage

The script `process_evaluation.py` calculates the Mean Absolute Error (MAE) and other statistics between model-generated answers and ground-truth answers. It expects both inputs to be in JSONL format.

### Input File Format

The script requires two files: one for predictions and one for ground truth. Both should be JSONL files, where each line is a JSON object.

Each JSON object must contain:
1.  A unique identifier key (e.g., `"image_id": "warehouse_sample_2.jpeg"`) to match predictions with ground truth.
2.  An `"answer"` key containing the text to be evaluated (e.g., `"answer": "The man is about 2 feet away."`).

The script automatically parses numerical values and common units (m, cm, ft) from the answer strings.

**Example `predictions.jsonl`:**
```json
{"image_id": "img_001.jpg", "question": "...", "answer": "<think>...</think> <answer>The person is approximately 1.5 meters from the table.</answer>"}
```

**Example `ground_truth.jsonl`:**
```json
{"image_id": "img_001.jpg", "question": "...", "answer": "1.4 meters"}
```

### Running the Evaluation

To run the evaluation, execute the script from the repository root:

```bash
python experiments/ravine_evaluator/process_evaluation.py \
    --predictions /path/to/your/predictions.jsonl \
    --ground-truth /path/to/your/ground_truth.jsonl \
    --id-key image_id
```

The script will print a report with the evaluation metrics to the console.
