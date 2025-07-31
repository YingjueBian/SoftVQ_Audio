#!/bin/bash

# Multi-GPU training script for AIMv2 Speech model

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on available GPUs
export OMP_NUM_THREADS=3
export TOKENIZERS_PARALLELISM=false

# Paths
CONFIG_PATH="configs/aimv2_speech_config.json"
OUTPUT_DIR="outputs/aimv2_speech_8gpu"
DEEPSPEED_CONFIG="configs/deepspeed_pretrain_config.json"

# Training parameters
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

echo "Starting AIMv2 Speech training with $NUM_GPUS GPUs"

# Create output directory
mkdir -p $OUTPUT_DIR

# DeepSpeed training
deepspeed train.py \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --deepspeed $DEEPSPEED_CONFIG

echo "Training completed!"