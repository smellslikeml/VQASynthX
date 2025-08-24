# Use a base image with PyTorch and CUDA.
# This aligns with the base images that could be used in the VQASynth project.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies needed for some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the evaluation script
# Using --no-cache-dir for a smaller image size
RUN pip install --no-cache-dir \
    "torch>=2.0" \
    "transformers>=4.36.0" \
    "pillow>=9.0.0" \
    "requests>=2.25.0" \
    "accelerate" \
    bitsandbytes

# Copy the evaluation script into the container
COPY run.py .

# Set the entrypoint to run the evaluation script
ENTRYPOINT ["python", "run.py"]
