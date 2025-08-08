# README for Temporal Ravine Evaluator

This directory contains a minimal, self-contained experiment to evaluate the temporal reasoning capabilities of `VQASynth`-trained models.

The evaluation is inspired by the sequence-based benchmarks from the `LTLZinc` repository, which generates tasks requiring reasoning over time. This experiment simplifies that concept into a "before-and-after" task, where a model must describe or answer questions about changes between two images.

## Usage

1. **Setup Environment**

   Ensure you have the main project dependencies installed. You can typically install them from the root `requirements.txt`. You will also need `Pillow`.
   ```bash
   pip install -r ../../requirements.txt
   pip install Pillow
   ```

2. **Hugging Face Login**

   You will need to authenticate to download the model from the Hugging Face Hub.
   ```bash
   huggingface-cli login
   ```

3. **Run the Evaluation**

   Execute the Python script. It will automatically load the `SpaceThinker` model, generate a toy dataset of two images in memory, and run the evaluation.

   ```bash
   python process_evaluation.py
   ```

## Expected Output

The script will print the model's response for each question in the evaluation set, whether the answer was considered correct based on keyword matching, and a final accuracy score at the end. This provides a baseline measurement for the model's ability to handle simple temporal queries.