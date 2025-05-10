#!/bin/bash

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

# Get the list of available GPUs
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)

# Function to run training on a specific GPU
run_training() {
    local gpu_id=$1
    local config_name=$2
    local embed_dim=$3
    local num_layers=$4
    local num_heads=$5
    local ff_dim=$6
    local max_seq_len=$7
    local dropout=$8
    local batch_size=$9
    local learning_rate=${10}

    echo "Starting training on GPU $gpu_id with config: $config_name"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python -m src.train_infilling_model \
        --model_dir "$MODEL_DIR/$config_name" \
        --data_dir "$DATA_DIR" \
        --cache_dir "$CACHE_DIR" \
        --tokenizer_model "gpt2" \
        --batch_size $batch_size \
        --num_epochs 10 \
        --learning_rate $learning_rate \
        --teacher_forcing_ratio 1.0 \
        --embed_dim $embed_dim \
        --num_layers $num_layers \
        --num_heads $num_heads \
        --ff_dim $ff_dim \
        --dropout $dropout \
        --max_seq_length $max_seq_len > "training_log_${config_name}_gpu${gpu_id}.log" 2>&1 &
}

# Read configurations from Python script
configs=$(python -c "from src.grid_search_config import generate_grid_search_configs, get_config_name; configs = generate_grid_search_configs(); print('\n'.join([f'{get_config_name(c)} {c.embed_dim} {c.num_layers} {c.num_heads} {c.ff_dim} {c.max_seq_len} {c.dropout} {c.batch_size} {c.learning_rate}' for c in configs]))")

# Counter for GPU assignment
gpu_counter=0

# Launch training jobs
while IFS= read -r line; do
    # Parse configuration
    read -r config_name embed_dim num_layers num_heads ff_dim max_seq_len dropout batch_size learning_rate <<< "$line"
    
    # Get GPU ID (round-robin assignment)
    gpu_id=$(echo "$GPUS" | sed -n "$((gpu_counter % 4 + 1))p")
    
    # Run training
    run_training "$gpu_id" "$config_name" "$embed_dim" "$num_layers" "$num_heads" "$ff_dim" "$max_seq_len" "$dropout" "$batch_size" "$learning_rate"
    
    # Increment counter
    ((gpu_counter++))
    
    # Wait a bit between launches to avoid overwhelming the system
    sleep 5
done <<< "$configs"

echo "All training jobs launched. Check individual log files for progress." 