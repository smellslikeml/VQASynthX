# Dockerfile for the YUME dynamic scene generation experiment
# This sets up the environment needed to run experiments/yume_sim/run.py

# Use the same base image as YUME for compatibility
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone the YUME source repository
RUN git clone https://github.com/stdstu12/YUME.git yume_repo

# 3. Install YUME's Python dependencies
# We move into the repo, install requirements, and then install the package itself
WORKDIR /app/yume_repo
RUN sed -i '/^torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# 4. Prepare for running the experiment script
WORKDIR /app
# Install requests for downloading the sample image in the run script
RUN pip install --no-cache-dir requests==2.31.0

# 5. Copy the experiment script into the container
# This assumes the Docker build context is the root of the experimental-vqasynth repo
COPY experiments/yume_sim/run.py .

# 6. Set the entrypoint to run our experiment script
ENTRYPOINT ["python", "run.py"]
