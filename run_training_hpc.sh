#!/bin/bash
#
# Script to run the TinyStories model training on HPC
#

# Create directories for data, models, and cache if they don't exist
HPC_BASE_DIR="/data/cmpe258-sp25/018219422"
mkdir -p "$HPC_BASE_DIR/data"
mkdir -p "$HPC_BASE_DIR/models"
mkdir -p "$HPC_BASE_DIR/cache"

# Set environment variables
export DATA_DIR="$HPC_BASE_DIR/data"
export MODEL_DIR="$HPC_BASE_DIR/models"
export CACHE_DIR="$HPC_BASE_DIR/cache"

# Make sure Python can find the modules
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Output info
echo "Running TinyStories training on HPC"
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Cache directory: $CACHE_DIR"
echo "Current directory: $(pwd)"

# Run the training script
python -m src.train_infilling_model \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$CACHE_DIR" \
    --tokenizer_model "gpt2" \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --teacher_forcing_ratio 1.0 \
    --embed_dim 384 \
    --num_layers 6 \
    --num_heads 6 \
    --ff_dim 1536 \
    --dropout 0.1

echo "Training completed!" 