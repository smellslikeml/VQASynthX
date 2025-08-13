# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# This container uses JAX. The PyTorch base image is a convenient
# source for a pre-configured CUDA development environment.
ENV CUDA_HOME=/usr/local/cuda-12.1

WORKDIR /app

# Install Python dependencies:
# 1. Install JAX with CUDA support.
# 2. Install the pobax library and its 'craftax' extra, which includes the
#    navix dependency required for the DMLab maze environments.
RUN pip install --no-cache-dir \
    "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "pobax[craftax]"

# Copy the experiment runner script into the container
COPY run.py .

# Run the benchmark experiment
CMD ["python", "run.py"]
