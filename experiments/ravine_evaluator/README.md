# Ravine Evaluator

This script is an experimental tool to evaluate the output of the VQASynth pipeline for its applicability to multi-agent cooperative scenarios. It is inspired by the evaluation methods used in multi-agent reinforcement learning (MARL) frameworks like MAGPO.

## Description

The evaluator processes the JSON output from the `scene_fusion_stage` of the VQASynth pipeline. It identifies objects that are likely to be 'agents' (e.g., persons, robots) and calculates the spatial distance between them.

A simple heuristic is applied: if two agents are within a predefined distance threshold, they are considered a 'successful' cooperative pair. The script aggregates these statistics across an entire dataset to produce a 'cooperation success rate', which measures how rich the dataset is with scenes depicting potential agent interaction.

This provides a quantitative metric to guide future dataset generation for multi-agent VQA.

## Usage

First, ensure you have a JSON file containing the fused scene data. This is typically the output of the VQASynth pipeline.

Then, run the script from the root of the repository:

```bash
# Example assumes the output of the VQASynth pipeline is in the cache.
python experiments/ravine_evaluator/process_evaluation.py \
    --input_data ./cache/your_dataset/scene_fusion_output.json \
    --output_metrics ./experiments/ravine_evaluator/metrics.json
```

The script will print the results to the console and save them to the specified output file.