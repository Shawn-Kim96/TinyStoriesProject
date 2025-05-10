#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --partition=lgpuq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --output=grid_search_%j.log
#SBATCH --error=grid_search_%j.err

# Load required modules
module load cuda/11.7
module load python/3.9

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

# Get node ID (0-3)
NODE_ID=$SLURM_NODEID
echo "Running on node $NODE_ID"

# Get the list of available GPUs for this node
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)

# Function to run training on a specific GPU
run_training() {
    local gpu_id=$1
    local config_name=$2
    local embed_dim=$3
    local num_layers=$4
    local num_heads=$5
    local ff_dim=$6
    local dropout=$7
    local learning_rate=$8

    echo "Node $NODE_ID: Starting training on GPU $gpu_id with config: $config_name"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python -m src.train_infilling_model \
        --model_dir "$MODEL_DIR/$config_name" \
        --data_dir "$DATA_DIR" \
        --cache_dir "$CACHE_DIR" \
        --tokenizer_model "gpt2" \
        --num_epochs 10 \
        --learning_rate $learning_rate \
        --teacher_forcing_ratio 1.0 \
        --embed_dim $embed_dim \
        --num_layers $num_layers \
        --num_heads $num_heads \
        --ff_dim $ff_dim \
        --dropout $dropout \
        > "training_log_node${NODE_ID}_${config_name}_gpu${gpu_id}.log" 2>&1 &
}

# Read configurations from Python script
configs=$(python -c "from src.grid_search_config import generate_grid_search_configs, get_config_name; configs = generate_grid_search_configs(); print('\n'.join([f'{get_config_name(c)} {c.embed_dim} {c.num_layers} {c.num_heads} {c.ff_dim} {c.dropout} {c.learning_rate}' for c in configs]))")

# Calculate which configurations this node should handle
total_configs=$(echo "$configs" | wc -l)
configs_per_node=$((total_configs / 4))
start_idx=$((NODE_ID * configs_per_node))
end_idx=$((start_idx + configs_per_node - 1))

# Get this node's configurations
node_configs=$(echo "$configs" | sed -n "$((start_idx + 1)),$((end_idx + 1))p")

# Counter for GPU assignment
gpu_counter=0

# Launch training jobs for this node
while IFS= read -r line; do
    # Parse configuration
    read -r config_name embed_dim num_layers num_heads ff_dim dropout learning_rate <<< "$line"
    
    # Get GPU ID (round-robin assignment)
    gpu_id=$(echo "$GPUS" | sed -n "$((gpu_counter % 4 + 1))p")
    
    # Run training
    run_training "$gpu_id" "$config_name" "$embed_dim" "$num_layers" "$num_heads" "$ff_dim" "$dropout" "$learning_rate"
    
    # Increment counter
    ((gpu_counter++))
    
    # Wait a bit between launches to avoid overwhelming the system
    sleep 5
done <<< "$node_configs"

# Wait for all background jobs to complete
wait

echo "Node $NODE_ID: All training jobs completed." 