# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the experiment files into the container
COPY ./experiments/maze_eval/ .

# Set the command to run the experiment script
CMD ["python", "run.py"]
