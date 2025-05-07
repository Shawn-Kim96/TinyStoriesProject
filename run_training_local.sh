#!/bin/bash

# Set base directories for local environment
BASE_DIR="$(pwd)"
DATA_DIR="$BASE_DIR/data"
MODEL_DIR="$BASE_DIR/models"
CACHE_DIR="$BASE_DIR/cache"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$CACHE_DIR"

# Set environment variables
export DATA_DIR
export MODEL_DIR
export CACHE_DIR

echo "Running TinyStories training in local environment"
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Cache directory: $CACHE_DIR"
echo "Current directory: $(pwd)"

# Run training script in online mode
python -m src.train_infilling_model \
    --data_dir="$DATA_DIR" \
    --model_dir="$MODEL_DIR" \
    --cache_dir="$CACHE_DIR" \
    --batch_size=32 \
    --num_epochs=10 \
    --learning_rate=1e-4 \
    --embed_dim=384 \
    --num_layers=6 \
    --num_heads=6 \
    --ff_dim=1536 \
    --dropout=0.1 \
    --log_interval=50 \
    --seed=42

echo "Training completed!" 