# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install Python dependencies
# google-generativeai for Gemini API access
# Pillow for image handling (dependency of the SDK for local files)
RUN pip install --no-cache-dir google-generativeai Pillow

# Copy the application scripts into the container
COPY process_ocr.py entrypoint.sh ./

# Make the entrypoint executable
RUN chmod +x entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["./entrypoint.sh"]