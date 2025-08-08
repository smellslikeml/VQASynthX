# Experiment: Reasoning Refinement with a Ravine Evaluator

This experiment introduces a post-processing step to the VQASynth pipeline, inspired by the iterative, critique-and-refine workflow from the ChatBattery project. The goal is to improve the quality and logical consistency of the generated reasoning traces (`<think>...</think>`).

## Description

The `process_evaluation.py` script simulates an evaluation and refinement loop. It takes a VQA sample (with potentially flawed reasoning) and uses a powerful LLM (e.g., GPT-4) as an "evaluator agent". This agent performs two steps:

1.  **Critique:** It analyzes the original reasoning for logical fallacies, unsupported assumptions, or irrelevant calculations.
2.  **Refine:** Based on its own critique, it generates a new, improved reasoning trace that is more sound and directly addresses the question.

This demonstrates a method for programmatic self-improvement of synthetic data, potentially leading to higher-quality training sets for spatial reasoning VLMs.

## Usage

### 1. Setup

Install the necessary Python library:
```bash
pip install openai
```

### 2. Set API Key

Export your OpenAI API key as an environment variable.
```bash
export OPENAI_API_KEY='your-key-here'
```

### 3. Run the Experiment

Execute the script:
```bash
python experiments/ravine_evaluator/process_evaluation.py
```

The script will print the original VQA pair, the full response from the evaluator agent (including the critique), and the final, parsed & refined reasoning block.