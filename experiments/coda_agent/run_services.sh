#!/usr/bin/env bash
set -euo pipefail

# This script sets up the inference services for the CODA dual-brain agent.
# It starts two VLLM servers: one for the planner model and one for the executor model.
# This provides the core components of the CODA agent for testing and integration.
#
# Based on the deployment strategy outlined in the CODA repository:
# https://github.com/OpenIXCLab/CODA
#
# Prerequisites:
# 1. A Python environment with CUDA support.
# 2. `vllm` and `huggingface-hub` installed:
#    pip install "vllm>=0.4.0" huggingface-hub
# 3. Logged into Hugging Face Hub with a token that has read access:
#    huggingface-cli login

# --- Configuration ---
PLANNER_MODEL="OpenIXCLab/CODA-PLANNER-TARS-32B"
EXECUTOR_MODEL="ByteDance-Seed/UI-TARS-1.5-7B"

PLANNER_PORT=${PLANNER_PORT:-8000}
EXECUTOR_PORT=${EXECUTOR_PORT:-8001}

# Using a tensor parallel size of 1 for a minimal, single-GPU setup.
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

# --- Service Deployment ---

echo "Starting CODA Planner service..."
echo "  - Model: ${PLANNER_MODEL}"
echo "  - Port: ${PLANNER_PORT}"
echo "  - TP Size: ${TENSOR_PARALLEL_SIZE}"

vllm serve "${PLANNER_MODEL}" \
    --served-model-name "coda-planner" \
    --host 0.0.0.0 \
    --port "${PLANNER_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" & 
PLANNER_PID=$!
echo "Planner service starting with PID ${PLANNER_PID}."

echo "-------------------------------------"
sleep 5

echo "Starting CODA Executor service..."
echo "  - Model: ${EXECUTOR_MODEL}"
echo "  - Port: ${EXECUTOR_PORT}"
echo "  - TP Size: ${TENSOR_PARALLEL_SIZE}"

vllm serve "${EXECUTOR_MODEL}" \
    --served-model-name "coda-executor" \
    --host 0.0.0.0 \
    --port "${EXECUTOR_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" &
EXECUTOR_PID=$!
echo "Executor service starting with PID ${EXECUTOR_PID}."

echo "-------------------------------------"

echo
echo "Both services are starting in the background."
echo "Planner URL: http://localhost:${PLANNER_PORT}"
echo "Executor URL: http://localhost:${EXECUTOR_PORT}"
echo
echo "To check service health (wait for models to load), run:"
echo "  curl http://localhost:${PLANNER_PORT}/health"
echo "  curl http://localhost:${EXECUTOR_PORT}/health"
echo
echo "To stop the services, run:"
echo "  kill ${PLANNER_PID} ${EXECUTOR_PID}"
echo

# Wait for any process to exit, then clean up others.
trap 'kill $(jobs -p)' EXIT
wait -n
