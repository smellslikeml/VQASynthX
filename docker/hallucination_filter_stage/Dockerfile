# syntax=docker/dockerfile:1.4
# Base image aligned with SOURCE repo environment for consistency
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies for the script
RUN pip install --no-cache-dir transformers==4.41.2 accelerate==0.30.1 tqdm

# Copy the processing script and its entrypoint
COPY process_hallucinations.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set Hugging Face cache to a shared volume to avoid re-downloading models
ENV HF_HOME=/data/cache

ENTRYPOINT ["./entrypoint.sh"]
