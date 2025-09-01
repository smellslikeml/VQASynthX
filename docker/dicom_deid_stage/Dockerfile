# syntax=docker/dockerfile:1.4
# Base image from the source project, proven to work with the dependencies.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
ENV CUDA_HOME=/usr/local/cuda-12.1

WORKDIR /app

# Install system dependencies for git and tesserocr (for presidio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libleptonica-dev \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the repository's scripts
RUN pip install --no-cache-dir \
    pydicom \
    pandas \
    matplotlib \
    presidio-image-redactor \
    tesserocr

# Copy the de-identification scripts and the entrypoint
COPY --chmod=755 deidentify_metadata.py .
COPY --chmod=755 redact_pixel_data.py .
COPY --chmod=755 entrypoint.sh .

ENTRYPOINT ["/app/entrypoint.sh"]