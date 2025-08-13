#!/bin/bash
set -e

# This is a minimal entrypoint for a single execution.
# In a full pipeline, this script would likely iterate over files in a mounted input directory.

echo "Running DCT feature extraction stage..."
python process_dct.py "$@"
echo "DCT feature extraction complete."
