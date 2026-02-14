# Use a slim python base image
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
# Using specific versions for reproducibility
RUN pip install pandas==2.0.3 scipy==1.11.2 plotly==5.15.0 kaleido==0.2.1

# Copy the analysis script into the container
COPY run_analysis.py .
COPY entrypoint.sh .

# Make the entrypoint executable
RUN chmod +x entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
