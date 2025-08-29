# Use a Pytorch base image consistent with the SOURCE repo environment.
# SOURCE_DOCKERFILE specifies torch 2.5.1 on CUDA 12.1 and python 3.10.
# We use a slim public image as a base.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install minimal dependencies. Unlike SOURCE, we don't need rdkit, etc.
RUN pip install tqdm==4.67.0

COPY synthesize_graphs.py ./
COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]