# syntax=docker/dockerfile:1.4

# This Dockerfile sets up an environment to run a POBAX RL experiment.
# It is based on the tested environment described in the POBAX source repository.
# It uses a PyTorch base image for its pre-installed CUDA development environment.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# POBAX uses JAX. The PyTorch base image provides a convenient CUDA env.
ENV CUDA_HOME=/usr/local/cuda-12.1

WORKDIR /app

# Install pobax and its dependencies, including JAX with CUDA support.
# The base image has CUDA 12.1, which is compatible.
# We install directly from PyPI as recommended for usage.
RUN pip install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --no-cache-dir pobax

# Copy the experiment runner script into the container's working directory.
COPY run.py .

# Execute the experiment script.
# The script calls the pobax PPO agent on T-Maze and leverages the GPU.
CMD ["python", "run.py"]
