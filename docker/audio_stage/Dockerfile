# syntax=docker/dockerfile:1.4
# Use the same base as the FISHER demo to ensure compatibility
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV CUDA_HOME=/usr/local/cuda-12.1
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir \
    einops \
    timm \
    omegaconf \
    torchaudio

# Set up the working directory and clone the FISHER source code
WORKDIR /app
RUN git clone https://github.com/jianganbai/FISHER.git .

# Download the pre-trained FISHER-tiny model checkpoint
RUN curl -L -o models/fisher-tiny.pt \
    'https://cloud.tsinghua.edu.cn/f/630a4b1b2962481a9150/?dl=1'

# Copy the processing script and entrypoint
COPY process_audio.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
