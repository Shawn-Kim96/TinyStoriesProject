# TinyStories Infilling Model

A language model trained on the TinyStories dataset that can generate the middle part of a story given the first and last sentences.

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Directory Structure

```
├── notebooks       # Jupyter notebooks for experiments
│   ├── 1.0-data-prep.ipynb
│   ├── 1.1-eda.ipynb
│   ├── 2.0-model-training.ipynb
│   ├── 3.0-model-evaluation.ipynb
│   ├── 4.0-visualization.ipynb
│
├── data            # Raw and processed datasets
│   ├── raw/
│   ├── processed/
│
├── models          # Saved models and checkpoints
│
├── src             # Python source scripts for automation
│   ├── data_extraction.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── reports         # Reports and visualizations
│
├── config          # Configuration files
│   ├── hyperparams.yaml
│
└── README.md       # Project documentation
```

## Git Workflow & Commit Message Convention

### Branching Strategy

- `main`: Stable and production-ready code.

- `feature/`: New features are developed here before merging into develop.

- `bugfix/`: Fixes for bugs found in develop.

- `hotfix/`: Urgent bug fixes that need to be patched into main.

### Commit Message Format

To maintain consistency and readability, we follow this structured format:
```
[type]: [short description]

[Optional detailed description]
```

**Commit Types**:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation updates
- `refactor`: Code restructuring without functionality changes
- `chore`: Maintenance tasks (build processes, dependency updates, etc.)