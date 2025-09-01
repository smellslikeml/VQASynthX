#!/usr/bin/env bash
set -euo pipefail

# This script demonstrates the PITA workflow adapted for VQA data synthesis.
# It assumes a pre-existing preference dataset in the format expected by the training script.

# --- Configuration ---
MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH="./dummy_vqa_preferences.jsonl" # Assumes a dummy dataset exists here
OUTPUT_DIR="./checkpoints/pita_vqa_classifier"
RESULTS_DIR="./results/pita_vqa_guided"

# --- Dummy Data Creation (for demonstration) ---
echo "Step 0/3: Creating dummy preference data..."
mkdir -p $(dirname "$DATA_PATH")
cat <<EOF > $DATA_PATH
{"question": "What is to the left of the plant?", "chosen": "To the left of the plant is a white couch.", "rejected": "A couch is there."}
{"question": "How far is the monitor from the keyboard?", "chosen": "The monitor is approximately 1 foot away from the keyboard on the desk.", "rejected": "The monitor is close to the keyboard."}
EOF
echo "Dummy data created at $DATA_PATH"

# --- Step 1: Train PITA Classifier ---
# This script adapts PITA's train_classifier.py to learn from VQA preference pairs.
# It fine-tunes a model to distinguish between 'chosen' and 'rejected' VQA answers.
echo "\nStep 1/3: Training VQA preference classifier..."
python experiments/pita_vqa/train_vqa_classifier.py \
  --model_id "$MODEL_ID" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs 1 \
  --batch_size 1 \
  --lr 2e-5

# --- Step 2: Guided VQA Generation ---
# This script adapts PITA's eval_ckpt.py to perform guided generation.
# It uses the trained classifier to steer the base model's output towards preferred responses.
# For this demo, we'll use the questions from our dummy dataset as prompts.
echo "\nStep 2/3: Running guided VQA generation..."
python experiments/pita_vqa/generate_guided_vqa.py \
  --base_model_id "$MODEL_ID" \
  --classifier_ckpt_path "${OUTPUT_DIR}/final_checkpoint/" \
  --prompts_path "$DATA_PATH" \
  --output_dir "$RESULTS_DIR" \
  --batch_size 1 \
  --max_new_tokens 50

# --- Step 3: Display Results ---
echo "\nStep 3/3: Experiment finished. Displaying results..."
cat "${RESULTS_DIR}/generated_vqa_responses.jsonl"
