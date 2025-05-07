#!/bin/bash
#
# Script to run the TinyStories model training on HPC
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

echo "Running TinyStories training on HPC"
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

# Use only first GPU
export CUDA_VISIBLE_DEVICES=0

# Try to free GPU memory before starting
python -c "import torch; torch.cuda.empty_cache()"

# Set memory allocation settings to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training script in single GPU mode with smaller parameters
python -m src.train_infilling_model \
    --data_dir="$DATA_DIR" \
    --model_dir="$MODEL_DIR" \
    --cache_dir="$CACHE_DIR" \
    --offline_mode \
    --batch_size=16 \
    --num_epochs=10 \
    --learning_rate=1e-4 \
    --embed_dim=128 \
    --num_layers=4 \
    --num_heads=4 \
    --ff_dim=256 \
    --dropout=0.1 \
    --log_interval=25 \
    --seed=42 \
    --num_workers=8 \
    --max_seq_length=128

echo "Training completed!"