# This Dockerfile encapsulates an experiment to generate dynamic video
# from a static image using the YUME model.
#
# Build command:
# docker build -t vqasynth-yume-exp -f experiments/yume_dynamic_scene_gen/Dockerfile .
#
# Run command (ensure you have an 'outputs' directory in your project root):
# mkdir -p outputs
# docker run --gpus all --rm -v $(pwd)/outputs:/app/outputs vqasynth-yume-exp

# Use a consistent base image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the YUME repository
RUN git clone https://github.com/stdstu12/YUME.git

WORKDIR /app/YUME

# Install Python dependencies from YUME's requirements
# Exclude torch/torchvision as they are in the base image
RUN sed -i '/^torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Download a sample image from the VQASynth repository to use as input
RUN mkdir -p ./input_image
RUN curl -L https://raw.githubusercontent.com/smellslikeml/experimental-vqasynth/main/assets/warehouse_sample_1.jpeg -o ./input_image/sample.jpg

# Create the caption file with a specific action prompt.
# This prompt instructs the model to generate a video of moving forward.
RUN echo "This video depicts a warehouse scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:5.0.Angular change rate (turn speed):1.0.View rotation speed:1.0." > ./caption.txt

# Run the inference script
# The output will be saved to the 'outputs' directory inside the container.
# We mount a local volume to this directory to retrieve the generated video.
CMD ["torchrun", "--nproc_per_node=1", "fastvideo/sample/sample.py", \
    "--video_output_dir=./outputs", \
    "--jpg_dir=./input_image", \
    "--caption_path=./caption.txt", \
    "--num_euler_timesteps=50", \
    "--seed=42", \
    "--mixed_precision=bf16", \
    "--allow_tf32", \
    "--t5_cpu"]
