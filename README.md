# TinyStories Project

## Overview
This project implements a story infilling model trained on the TinyStories dataset. The model is designed to generate text that fills in the middle of a story given its beginning and end.

## Project Structure
```
TinyStoriesProject/
├── data/
│   └── tinystories_data/
│       ├── train/
│       └── validation/
├── models/
│   └── tinystories_bpe_infilling_model.pth
├── cache/
│   └── tokenizers/
│       └── gpt2/
│           └── gpt2_tokenizer/
│               └── models--gpt2/
│                   └── snapshots/
│                       └── 607a30d783dfa663caf39e06633721c8d4cfcd7e/
│                           ├── config.json
│                           ├── merges.txt
│                           ├── tokenizer_config.json
│                           ├── tokenizer.json
│                           └── vocab.json
├── src/
│   ├── train_infilling_model.py
│   ├── models.py
│   ├── dataset.py
│   ├── bpe_tokenizer.py
│   └── config.py
├── run_training_hpc.sh
└── run_training_local.sh
```

## Setup

### Prerequisites
- Python 3.11+
- PyTorch
- Transformers
- CUDA-capable GPU (for training)

### Installation
1. Create a conda environment:
```bash
conda create -n tinystories python=3.11
conda activate tinystories
```

2. Install required packages:
```bash
pip install torch transformers datasets pyarrow
```

## Online vs Offline Environments

This project is designed to work in both online (with internet access) and offline (no internet access) environments:

- **Online Mode**: Directly downloads and uses the TinyStories dataset and GPT-2 tokenizer from Hugging Face.
- **Offline Mode**: Uses pre-downloaded datasets and tokenizer files stored locally.

### Local Development (Online Mode)
In an environment with internet access, you can run the model without pre-downloading any files:

1. Make the training script executable:
```bash
chmod +x run_training_local.sh
```

2. Run the training script:
```bash
./run_training_local.sh
```

This script will:
- Automatically download the TinyStories dataset from Hugging Face
- Download and cache the GPT-2 tokenizer
- Train the model using these resources
- Save the trained model to the `models` directory

### HPC Environment (Offline Mode)
For environments without internet access (like HPC clusters), you need to prepare the data beforehand:

1. **On a machine with internet access:**
   - Download the dataset and tokenizer as explained in the "Offline Environment Setup" section below.
   - Transfer the files to the HPC environment.

2. **On the HPC environment:**
   - Make the training script executable:
   ```bash
   chmod +x run_training_hpc.sh
   ```
   
   - Run the training script:
   ```bash
   ./run_training_hpc.sh
   ```

The HPC script includes the `--offline_mode` flag to ensure the model uses locally stored files without attempting to connect to the internet.

## Offline Environment Setup
To prepare files for an offline environment:

1. Download the TinyStories dataset on a machine with internet access:
```bash
# Option 1: Using datasets library
python -c "from datasets import load_dataset; dataset = load_dataset('roneneldan/TinyStories'); dataset['train'].to_json('data/tinystories_data/train.jsonl'); dataset['validation'].to_json('data/tinystories_data/validation.jsonl')"

# Option 2: Using datasets library with save_to_disk (Arrow format)
python -c "from datasets import load_dataset; dataset = load_dataset('roneneldan/TinyStories'); dataset.save_to_disk('data/tinystories_data')"

# Option 3: Using Hugging Face CLI
huggingface-cli download roneneldan/TinyStories --local-dir ./data/tinystories_data
```

2. Download and cache the GPT-2 tokenizer:
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2', cache_dir='cache/tokenizers/gpt2')"
```

3. Transfer the files to the offline environment:
```bash
# Transfer dataset
scp -r data/tinystories_data username@hpc-host:/data/cmpe258-sp25/your-username/data/

# Transfer tokenizer files
scp -r cache/tokenizers/gpt2 username@hpc-host:/data/cmpe258-sp25/your-username/cache/tokenizers/
```

4. Verify the correct directory structure on the HPC:
```
/data/cmpe258-sp25/your-username/
├── data/
│   └── tinystories_data/
│       ├── train/
│       └── validation/
└── cache/
    └── tokenizers/
        └── gpt2/
            └── gpt2_tokenizer/
                └── models--gpt2/
                    └── snapshots/
                        └── 607a30d783dfa663caf39e06633721c8d4cfcd7e/
                            ├── config.json
                            ├── merges.txt
                            ├── tokenizer_config.json
                            ├── tokenizer.json
                            └── vocab.json
```

## Training

### Training Parameters
Both training scripts use the following default parameters:
- Batch size: 32
- Number of epochs: 10
- Learning rate: 1e-4
- Model architecture:
  - Embedding dimension: 384
  - Number of layers: 6
  - Number of attention heads: 6
  - Feed-forward dimension: 1536
  - Dropout: 0.1

### Customizing Training
You can customize the training by modifying the training scripts or by running the training module directly:

```bash
python -m src.train_infilling_model \
  --data_dir="./data" \
  --model_dir="./models" \
  --cache_dir="./cache" \
  --offline_mode \  # Add this flag for offline mode
  --batch_size=64 \
  --num_epochs=20 \
  --learning_rate=5e-5 \
  # Add other parameters as needed
```

## Troubleshooting

### Tokenizer Loading Issues
If you encounter issues loading the tokenizer:

1. Verify the tokenizer directory structure:
```bash
ls -R /data/cmpe258-sp25/your-username/cache/tokenizers/gpt2/
```

2. Make sure the `offline_mode` parameter is set correctly:
   - For offline environments: `--offline_mode` flag should be used
   - For online environments: omit the `--offline_mode` flag

### Dataset Loading Issues
If you have issues loading the dataset:

1. For offline mode, verify the dataset structure:
```bash
ls -R /data/cmpe258-sp25/your-username/data/tinystories_data/
```

2. For online mode, check your internet connection and Hugging Face API access.

## Model Architecture
The model uses a transformer-based architecture with:
- BPE tokenization using GPT-2 tokenizer
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization and residual connections

## Recent Updates
- Added separate scripts for online and offline environments
- Implemented automatic dataset loading based on environment
- Updated tokenizer to support both online and offline modes
- Enhanced error handling for both modes
- Added detailed documentation for different usage scenarios

## Notes
- The model is designed to work in both online and offline environments
- For offline use, all necessary files must be downloaded and placed in the correct directories before training
- The model automatically adapts to the specified environment

## Git Workflow & Commit Message Convention

### Branching Strategy
- `main`: Stable and production-ready code
- `feature/`: New features
- `bugfix/`: Bug fixes
- `hotfix/`: Urgent bug fixes

### Commit Message Format
```
[type]: [short description]

[Optional detailed description]
```

**Commit Types**:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation updates
- `refactor`: Code restructuring
- `chore`: Maintenance tasks

## Project Overview

This project implements a transformer-based language model that performs story infilling on the TinyStories dataset. Given the first and last sentences of a story, the model generates the missing middle part, creating a complete coherent narrative.

### Features

- Transformer-based architecture for story infilling
- Training script with customizable parameters
- Inference script for generating stories
- Interactive notebook for exploring the model

## Setup

### Prerequisites

This project requires Python 3.11+ and the following libraries:
- PyTorch
- Datasets (HuggingFace)
- Transformers
- Pandas
- Scikit-learn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TinyStoriesProject.git
cd TinyStoriesProject
```

2. Install dependencies using Poetry:
```bash
poetry install
```

Or with pip:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:

```bash
python src/train_infilling_model.py --num_epochs 10 --batch_size 32
```

You can customize training parameters:
```bash
python src/train_infilling_model.py \
  --vocab_size 30000 \
  --max_seq_length 256 \
  --embed_dim 384 \
  --num_layers 6 \
  --num_heads 6 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_epochs 10
```

### Generating Stories

Once the model is trained, you can generate stories using:

```bash
python src/generate_story.py \
  --first_sentence "Once upon a time, there was a little boy named Tim who loved to play with toys." \
  --last_sentence "Tim learned that sharing his toys made everyone happy, including himself." \
  --max_tokens 200 \
  --temperature 1.0
```

### Interactive Demo

You can also explore the model using the provided Jupyter notebook:

```bash
jupyter notebook notebooks/story_infilling_demo.ipynb
```

## Model Architecture

The model uses a transformer-based architecture:

- Embedding layer: Converts token IDs to dense vectors
- Positional encoding: Adds position information to embeddings
- Transformer layers: Self-attention and feed-forward networks
- Output layer: Projects to vocabulary size

The model is trained to predict the full story given the first and last sentences separated by a special `<blank>` token.

## Dataset

The model is trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), which contains simple short stories designed for language model training. The stories are split into first sentence, middle part, and last sentence during training.

## Results

The model learns to generate coherent middle parts of stories that connect the first and last sentences. It captures the narrative structure of children's stories and can produce creative and contextually appropriate content.
