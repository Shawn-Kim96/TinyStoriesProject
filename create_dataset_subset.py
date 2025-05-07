#!/usr/bin/env python
# Script to create a subset of the TinyStories dataset

import json
import argparse
from pathlib import Path
import random

def create_dataset_subset(input_dir, output_dir, split, num_samples, seed=42):
    """
    Create a smaller subset of the TinyStories dataset.
    
    Args:
        input_dir: Path to the original dataset directory
        output_dir: Path to save the subset
        split: Dataset split (train or validation)
        num_samples: Number of samples to include in the subset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_file = Path(input_dir) / f"{split}.jsonl"
    output_file = Path(output_dir) / f"{split}.jsonl"
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating subset of {num_samples} samples from {input_file}")
    
    # Load all data
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(line)
    
    total_samples = len(all_data)
    print(f"Loaded {total_samples} samples from original dataset")
    
    # Sample subset
    if num_samples >= total_samples:
        subset = all_data
        print(f"Requested more samples than available, using all {total_samples} samples")
    else:
        subset = random.sample(all_data, num_samples)
        print(f"Randomly sampled {num_samples} examples")
    
    # Write subset to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in subset:
            f.write(line)
    
    print(f"Subset saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create a subset of TinyStories dataset")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the original TinyStories dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save the subset dataset')
    parser.add_argument('--num_samples', type=int, required=True,
                        help='Number of samples to include in the subset')
    parser.add_argument('--splits', type=str, default="train,validation",
                        help='Comma-separated list of dataset splits to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    for split in args.splits.split(','):
        create_dataset_subset(
            args.input_dir,
            args.output_dir,
            split,
            args.num_samples,
            args.seed
        )

if __name__ == "__main__":
    main() 