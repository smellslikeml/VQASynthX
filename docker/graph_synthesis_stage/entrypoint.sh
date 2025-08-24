#!/usr/bin/env bash
set -euo pipefail

echo "Starting SparsePGC Scene Graph Synthesis Experiment..."

# Execute the main python script
# The --gpus all flag should be supplied by the docker run command if available.
python synthesize_graphs.py

echo
echo "Experiment finished successfully."