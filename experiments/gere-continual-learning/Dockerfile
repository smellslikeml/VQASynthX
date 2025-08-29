# Base image matching GeRe's recommended torch version
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Install git and other basics
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all experiment files into the container
COPY . .

# Install Python dependencies
# These are a combination of requirements from GeRe and typical VLM finetuning.
RUN pip install --no-cache-dir \
    transformers==4.44.2 \
    tokenizers==0.19.1 \
    accelerate==0.30.1 \
    deepspeed==0.14.4 \
    datasets \
    tqdm

# Set the command to run the training script
CMD ["python", "train.py"]
