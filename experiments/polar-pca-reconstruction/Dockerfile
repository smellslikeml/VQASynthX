# Use a standard Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the entire repository content into the image
# Assumes the build context is the repository root
COPY . .

# Install dependencies required for this specific experiment
# Using opencv-python-headless for a non-GUI environment
RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    opencv-python-headless \
    matplotlib \
    Pillow

# Set matplotlib backend to a non-interactive one to prevent errors in docker
ENV MPLBACKEND=agg

# Run the experiment script
CMD ["python", "experiments/polar-pca-reconstruction/run.py"]
