#!/bin/sh
set -e

# This script would be called by the main pipeline runner (e.g., run.sh)
# It expects environment variables for input and output paths.

INPUT_PATH=${INPUT_DIR}/prompt_data.jsonl
OUTPUT_PATH=${OUTPUT_DIR}/validated_prompt_data.jsonl

echo "Starting Validation and Refinement Stage..."
echo "Input file: ${INPUT_PATH}"
echo "Output file: ${OUTPUT_PATH}"

python process_validation.py --input "${INPUT_PATH}" --output "${OUTPUT_PATH}"

echo "Validation and Refinement Stage complete."