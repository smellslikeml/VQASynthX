# syntax=docker/dockerfile:1.4

FROM python:3.9-slim

WORKDIR /app

COPY process_validation.py .
COPY entrypoint.sh .

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]