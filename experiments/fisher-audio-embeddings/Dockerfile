# syntax=docker/dockerfile:1.4

# Use the same PyTorch base image as the FISHER demo to ensure compatibility.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install git to clone the repository.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the FISHER repository which contains the model definition.
# We need this to import `models.fisher`
RUN git clone https://github.com/jianganbai/FISHER.git fisher

# Set PYTHONPATH to include the cloned FISHER directory.
ENV PYTHONPATH=/app/fisher

# Download the pre-trained FISHER model checkpoint.
RUN mkdir -p /app/models && \
    curl -L -o /app/models/fisher-tiny.pt \
    'https://cloud.tsinghua.edu.cn/f/630a4b1b2962481a9150/?dl=1'

# Copy the experiment script into the container.
COPY run.py /app/run.py

# Run the experiment script.
CMD ["python", "/app/run.py"]
