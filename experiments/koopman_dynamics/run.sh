#!/usr/bin/env bash

set -euo pipefail

# Ensure python and pip are available
if ! command -v python &> /dev/null
then
    echo "python could not be found, please install it."
    exit 1
fi

# Install dependencies
if [ -f "requirements.txt" ]; then
    pip install --progress-bar off -r requirements.txt
fi

# Run the experiment
python run.py
