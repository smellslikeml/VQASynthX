# Ravine Evaluator

This script provides a minimal framework for quantitatively evaluating the performance of VLMs on metric distance estimation tasks. It is inspired by the structured benchmarking approach found in libraries like `confopt` and is intended to provide a systematic way to compare models trained using `VQASynth` datasets.

## Purpose

The primary goal of `VQASynth` is to enhance the spatial reasoning of VLMs. This script offers a concrete way to measure one key aspect of that enhancement: the accuracy of distance predictions. By running a model against a dataset with ground-truth distances, we can calculate a Mean Absolute Error (MAE) metric, allowing for objective model comparison.

## Evaluation Data Format

The script expects a JSON Lines (`.jsonl`) file where each line is a JSON object with the following keys:

- `image_path`: A string containing the local path or a public URL to an image.
- `prompt`: The question to ask the model. It should explicitly ask for a distance. For best results with the current script, prompt the model to answer in meters.
- `ground_truth_meters`: A floating-point number representing the true distance in meters.

### Example `eval_data.jsonl`

```json
{"image_path": "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_2.jpeg?raw=true", "prompt": "How many meters away is the man in the red hat from the wooden pallet?", "ground_truth_meters": 0.6}
{"image_path": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat-cafe.png", "prompt": "Estimate the distance in meters between the two cats on the floor.", "ground_truth_meters": 1.2}
```

## Usage

1.  **Install Dependencies**: Ensure you have `torch`, `transformers`, `Pillow`, and `requests` installed in your environment.

    ```bash
    pip install torch transformers Pillow requests
    ```

2.  **Prepare Data**: Create an evaluation file in the format described above.

3.  **Run Evaluation**: Execute the script from the repository root, pointing to your model and evaluation file.

    ```bash
    python experiments/ravine_evaluator/process_evaluation.py \
      --model_id remyxai/SpaceThinker-Qwen2.5VL-3B \
      --eval_file path/to/your/eval_data.jsonl
    ```

## Output

The script will print the model's response and the calculated error for each sample, followed by a final summary.

```
--- Processing Sample 1 ---
Prompt: How many meters away is the man in the red hat from the wooden pallet?
Model Response: <think>...</think> <answer>The man in the red hat is approximately 0.6 meters from the wooden pallet.</answer>
Ground Truth: 0.60m | Predicted: 0.60m | Absolute Error: 0.00m

...

--- Evaluation Summary ---
Total samples processed successfully: 2
Mean Absolute Error (MAE): 0.0512 meters
```