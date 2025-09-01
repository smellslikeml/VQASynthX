# syntax=docker/dockerfile:1.4

# Use a base image with PyTorch and CUDA
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install git and other essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir transformers==4.40.1 accelerate==0.29.3 sentencepiece==0.2.0

# Copy the processing script and entrypoint
COPY process_reasoning_rewards.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]