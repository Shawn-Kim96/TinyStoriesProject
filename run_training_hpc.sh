#!/bin/bash
#
# Script to run the TinyStories model training on HPC
#

# Create directories for data, models, and cache if they don't exist
HPC_BASE_DIR="/data/cmpe258-sp25/018219422"
mkdir -p "$HPC_BASE_DIR/data"
mkdir -p "$HPC_BASE_DIR/models"
mkdir -p "$HPC_BASE_DIR/cache"
mkdir -p "$HPC_BASE_DIR/cache/tokenizers/gpt2"

# Set environment variables
export DATA_DIR="$HPC_BASE_DIR/data"
export MODEL_DIR="$HPC_BASE_DIR/models"
export CACHE_DIR="$HPC_BASE_DIR/cache"
export TRANSFORMERS_CACHE="$HPC_BASE_DIR/cache"
export HF_HOME="$HPC_BASE_DIR/cache"
export HF_DATASETS_CACHE="$HPC_BASE_DIR/cache/datasets"

# Make sure Python can find the modules
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Output info
echo "Running TinyStories training on HPC"
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Cache directory: $CACHE_DIR"
echo "Current directory: $(pwd)"

# Check for required files for the tokenizer
if [ ! -d "$HPC_BASE_DIR/cache/tokenizers/gpt2/models--gpt2" ]; then
    echo "Tokenizer files not found. You may need to manually copy them from your local machine."
    echo "Use: scp -r /path/to/local/cache/tokenizers/gpt2/models--gpt2 username@coe-hpc1.sjsu.edu:$HPC_BASE_DIR/cache/tokenizers/gpt2/"
    echo "Continuing anyway, but may fail if offline mode is enabled."
fi

# Run the training script with the specified arguments
python -m src.train_infilling_model_with_encoder \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$CACHE_DIR" \
    --tokenizer_model "gpt2" \
    "$@"  # Pass all command line arguments

echo "Training completed!"