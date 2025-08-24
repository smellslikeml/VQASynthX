#!/bin/bash
set -e

# These variables are expected to be set by an orchestration script (e.g., run.sh).
# Default values are provided for clarity and to enable standalone testing.
# The paths assume a /data volume is mounted into the container.
INPUT_FILE=${INPUT_FILE:-"/data/prompt_stage/output.jsonl"}
OUTPUT_FILE=${OUTPUT_FILE:-"/data/hallucination_filter_stage/filtered_output.jsonl"}
MODEL_NAME=${MODEL_NAME:-"mistralai/Mistral-7B-v0.1"}
ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:-"1.5"}

echo "--- Running Hallucination Filter Stage ---"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Model for logit reproduction: $MODEL_NAME"
echo "First-token entropy threshold: $ENTROPY_THRESHOLD"

# Execute the processing script with the provided arguments
python /app/process_hallucinations.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_name "$MODEL_NAME" \
    --entropy_threshold "$ENTROPY_THRESHOLD"

echo "--- Hallucination Filter Stage Complete ---"
