#!/bin/bash
#
# Script to run the TinyStories model training on multiple GPUs
#

# Set base directory for HPC environment
HPC_BASE_DIR="/data/cmpe258-sp25/018219422"

# Create directories if they don't exist
DATA_DIR="$HPC_BASE_DIR/data"
MODEL_DIR="$HPC_BASE_DIR/models"
CACHE_DIR="$HPC_BASE_DIR/cache"

mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$CACHE_DIR"

# Set environment variables
export DATA_DIR
export MODEL_DIR
export CACHE_DIR

echo "Running TinyStories multi-GPU training"
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Cache directory: $CACHE_DIR"
echo "Current directory: $(pwd)"

# Check if dataset exists
DATASET_DIR="$DATA_DIR/tinystories_data"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found at $DATASET_DIR"
    echo "Please download the dataset first or check the path."
    exit 1
fi

if [ ! -d "$DATASET_DIR/train" ] && [ ! -f "$DATASET_DIR/train.jsonl" ]; then
    echo "Error: Train dataset not found in $DATASET_DIR"
    echo "Please ensure either train.jsonl or train/ directory exists."
    exit 1
fi

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of available GPUs: $NUM_GPUS"

# Try to free GPU memory before starting
python -c "import torch; torch.cuda.empty_cache()"

# Set memory allocation settings to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set NCCL environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

# Run training script with distributed mode enabled
python -m src.parallel_train_infilling_model \
    --data_dir="$DATA_DIR" \
    --model_dir="$MODEL_DIR" \
    --cache_dir="$CACHE_DIR" \
    --offline_mode \
    --distributed \
    --batch_size=16 \
    --num_epochs=10 \
    --learning_rate=1e-4 \
    --embed_dim=256 \
    --num_layers=6 \
    --num_heads=8 \
    --ff_dim=512 \
    --dropout=0.1 \
    --log_interval=25 \
    --seed=42 \
    --num_workers=4

echo "Multi-GPU training completed!"