# Ravine: LLM-based Evaluator for Spatial Reasoning

Ravine is an experimental, automated evaluation pipeline for benchmarking Vision Language Models (VLMs) on spatial reasoning tasks. It is inspired by the LLM-as-a-judge methodology proposed in the [EESE](https://github.com/aiben-ch/EESE) project.

The script `process_evaluation.py` automates the following process:
1.  Loads a VLM and an evaluation dataset from the Hugging Face Hub.
2.  For each sample, it generates an answer from the VLM.
3.  It then uses a powerful "judge" LLM (like GPT-4o) to compare the VLM's answer to the ground truth and assign a score from 0 to 10.
4.  Results, including the question, answers, and score, are saved to a JSONL file.

## Setup

### 1. Dependencies

Install the required Python packages:

```bash
pip install transformers torch Pillow openai requests datasets
```

### 2. API Keys

The script requires API keys for OpenAI (for the judge model) and optionally for Hugging Face (to access gated models/datasets). Set them as environment variables:

```bash
export OPENAI_API_KEY='your-openai-api-key'
export HUGGINGFACE_TOKEN='your-huggingface-read-token'
```

## Usage

Run the evaluation script from the root of the repository. You must specify the model to test and the dataset to use.

```bash
python experiments/ravine_evaluator/process_evaluation.py \
    --model_id "remyxai/SpaceThinker-Qwen2.5VL-3B" \
    --dataset_id "remyxai/OpenSpaces_MC_R1" \
    --judge_model "gpt-4o" \
    --output_file "results/qwen_evaluation.jsonl" \
    --num_samples 50
```

**Arguments:**
-   `--model_id`: (Required) The Hugging Face Hub ID of the VLM to evaluate.
-   `--dataset_id`: (Required) The Hugging Face Hub ID of the evaluation dataset.
-   `--judge_model`: The model to use for judging. Defaults to `gpt-4o`.
-   `--output_file`: Path to save the output results. Defaults to `evaluation_results.jsonl`.
-   `--num_samples`: The number of samples to process from the dataset. Defaults to `20`.

## Output Format

The output is a JSONL file where each line is a JSON object with the following structure:

```json
{
  "id": 0,
  "question": "How far is the chair from the bookshelf in meters?",
  "ground_truth_answer": "The chair is approximately 1.5 meters from the bookshelf.",
  "model_answer": "The distance between the chair and the bookshelf is about 1.4 meters.",
  "score": 9
}
```
