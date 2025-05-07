#!/bin/bash
# Script to create a subset of the TinyStories dataset and run training on it

# Default values
SUBSET_SIZE=10000
BATCH_SIZE=16
NUM_EPOCHS=10
LEARNING_RATE=1e-4
EMBED_DIM=128
NUM_LAYERS=4
NUM_HEADS=4
FF_DIM=256
DROPOUT=0.1
LOG_INTERVAL=25
NUM_WORKERS=8
MAX_SEQ_LENGTH=128
MIN_STORY_LENGTH=5
DATA_DIR="/data/cmpe258-sp25/018219422/data"
MODEL_DIR="/data/cmpe258-sp25/018219422/models"
CACHE_DIR="/data/cmpe258-sp25/018219422/cache"
SUBSET_DIR="/data/cmpe258-sp25/018219422/data/tinystories_subset"
DOWNLOAD_TOKENIZER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --subset_size)
      SUBSET_SIZE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --model_dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --cache_dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --subset_dir)
      SUBSET_DIR="$2"
      shift 2
      ;;
    --download_tokenizer)
      DOWNLOAD_TOKENIZER=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Download tokenizer if required
if [ "$DOWNLOAD_TOKENIZER" = true ]; then
  echo "Downloading tokenizer for offline use..."
  python -m download_tokenizer --cache_dir="${CACHE_DIR}"
fi

echo "Creating subset of TinyStories dataset with $SUBSET_SIZE samples"
python -m create_dataset_subset \
  --input_dir="${DATA_DIR}/tinystories_data" \
  --output_dir="${SUBSET_DIR}" \
  --num_samples="${SUBSET_SIZE}" \
  --splits="train,validation" \
  --seed=42

echo "Running training on subset dataset"
python -m src.train_infilling_model \
  --data_dir="${DATA_DIR}" \
  --model_dir="${MODEL_DIR}/subset_${SUBSET_SIZE}" \
  --cache_dir="${CACHE_DIR}" \
  --offline_mode \
  --tokenizer_model="gpt2" \
  --batch_size="${BATCH_SIZE}" \
  --num_epochs="${NUM_EPOCHS}" \
  --learning_rate="${LEARNING_RATE}" \
  --embed_dim="${EMBED_DIM}" \
  --num_layers="${NUM_LAYERS}" \
  --num_heads="${NUM_HEADS}" \
  --ff_dim="${FF_DIM}" \
  --dropout="${DROPOUT}" \
  --log_interval="${LOG_INTERVAL}" \
  --seed=42 \
  --num_workers="${NUM_WORKERS}" \
  --max_seq_length="${MAX_SEQ_LENGTH}" \
  --min_story_length="${MIN_STORY_LENGTH}" \
  --max_samples="${SUBSET_SIZE}"

echo "Training completed!" 