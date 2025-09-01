#!/bin/bash
set -e

# Check if the required GOOGLE_API_KEY is provided.
if [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "Error: GOOGLE_API_KEY environment variable is not set." >&2
  echo "Please run the container with: -e GOOGLE_API_KEY=<your_key>" >&2
  exit 1
fi

# Execute the main script with all provided arguments.
exec python /app/process_ocr.py "$@"