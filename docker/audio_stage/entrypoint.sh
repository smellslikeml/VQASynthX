#!/bin/bash
set -e

# Expects arguments to be passed from the docker run command
# Example: --input_path /data/audio.wav --output_path /data/audio_embedding.pt
exec python /app/process_audio.py "$@"
