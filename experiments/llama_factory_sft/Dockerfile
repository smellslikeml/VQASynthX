# Start from a base image that LLaMA-Factory is compatible with.
# The source README indicates PyPI availability and wide model support.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install git and other essentials
RUN apt-get update && apt-get install -y --no-install-recommends git wget && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install LLaMA-Factory and its VLM dependencies
# The README highlights its wide support and PyPI package
RUN pip install --no-cache-dir "llamafactory[vlm]"

# Copy the VQASynth project code into the container
COPY ../../. .

# Install VQASynth dependencies
# This makes the VQASynth modules available if needed,
# though this experiment only uses its data.
RUN pip install --no-cache-dir -r requirements.txt

# --- This experiment does not require running the full VQASynth pipeline ---
# The run.py script will pull the pre-generated dataset from Hugging Face Hub.

# Set the entrypoint for running the experiment
CMD ["python", "experiments/llama_factory_sft/run.py"]
