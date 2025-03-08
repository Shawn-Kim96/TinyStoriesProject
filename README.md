# TinyStoriesProject
Tiny Stories Project

## Project Summary

Develop a NLP model inspired by [TinyStories](https://arxiv.org/abs/2305.07759).

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