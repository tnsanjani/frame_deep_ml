#!/bin/bash
# Environment setup
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Launch training with Accelerate
accelerate launch --multi_gpu --num_processes 7 train.py
