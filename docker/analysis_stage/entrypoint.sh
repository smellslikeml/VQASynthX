#!/bin/bash
# Default paths
INPUT_DIR="/data/input"
OUTPUT_DIR="/data/output"

# Execute the python script with the provided directories
python /app/run_analysis.py --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
