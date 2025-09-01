#!/bin/bash
set -e

# Expects environment variables to be set by docker run command
# - INPUT_PATH: Path to the input JSONL file inside the container
# - OUTPUT_PATH: Path to write the output JSONL file inside the container
# - CQT_FIELD: The key in the JSON object that contains the CoT string.
# - QUESTION_FIELD: The key in the JSON object that contains the question string.

# Default values
MODEL_ID_PRM=${MODEL_ID_PRM:="UW-Madison-Lee-Lab/VersaPRM"}
BATCH_SIZE=${BATCH_SIZE:=8}

echo "Starting reasoning reward processing..."

python process_reasoning_rewards.py \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --cot_field "${COT_FIELD}" \
    --question_field "${QUESTION_FIELD}" \
    --model_id_prm "${MODEL_ID_PRM}" \
    --batch_size "${BATCH_SIZE}"

echo "Reasoning reward processing complete. Output at ${OUTPUT_PATH}"