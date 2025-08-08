# Ravine Evaluation Harness

This script evaluates the performance of a VQASynth-trained Vision Language Model (VLM) on a custom spatial reasoning benchmark. It is designed to be a minimal, reusable tool for quantitative assessment.

The primary metric calculated is the Mean Absolute Error (MAE) for distance estimation tasks. The script parses numeric values and common units (e.g., meters, cm, feet) from both the model's generated answer and the ground truth, converts them to meters, and computes the error.

## Setup

Install the necessary dependencies:

```bash
pip install transformers torch Pillow requests numpy tqdm accelerate
```

## Data Format

The evaluation script expects a `.jsonl` file where each line is a JSON object with the following keys:

- `image`: A URL or a local file path to the image.
- `question`: The spatial reasoning question to ask the model.
- `ground_truth_answer`: A string containing the correct answer. For distance questions, this must include a number and a unit (e.g., "The forklift is 3.5 meters away.").

**Example `eval_data.jsonl`:**

```json
{"image": "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_2.jpeg?raw=true", "question": "How far is the man in the red hat from the pallet of boxes in meters?", "ground_truth_answer": "The man is about 0.6 meters from the pallet."}
{"image": "path/to/your/local/image.jpg", "question": "What is the distance between the two cars in feet?", "ground_truth_answer": "They are approximately 15 feet apart."}
```

## Usage

Run the evaluation from the command line, specifying the model to test and your evaluation data file.

```bash
python experiments/ravine_evaluator/process_evaluation.py \
    --model_id remyxai/SpaceThinker-Qwen2.5VL-3B \
    --eval_file ./eval_data.jsonl \
    --output_file ./results/spacetinker_results.jsonl
```

The script will print a summary of the results, including the Mean Absolute Error, to the console and save detailed per-item results to the specified output file.