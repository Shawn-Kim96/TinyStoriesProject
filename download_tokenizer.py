#!/usr/bin/env python
# Script to download tokenizer for offline use

import argparse
from transformers import AutoTokenizer
from pathlib import Path
import os
from src.config import get_default_tokenizer_model, get_cache_dir

def download_tokenizer(model_name, cache_dir):
    """
    Download and cache a tokenizer for offline use.
    
    Args:
        model_name: Name of the pre-trained model to download tokenizer for
        cache_dir: Directory to save cached files
    """
    # Configure paths according to Hugging Face cache structure
    base_cache = Path(cache_dir) / "tokenizers" / model_name
    
    print(f"Downloading tokenizer from '{model_name}'...")
    print(f"Cache directory: {base_cache}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=str(base_cache),
            trust_remote_code=True
        )
        print(f"Successfully downloaded tokenizer: {tokenizer.__class__.__name__}")
        
        # Add some special tokens to ensure they're downloaded too
        tokenizer.add_special_tokens({"additional_special_tokens": ["<blank>"]})
        
        # Print cache structure for reference
        tokenizer_cache = base_cache / f"models--{model_name.replace('/', '--')}"
        
        print("\nCache structure created:")
        print(f"Base cache: {base_cache}")
        if tokenizer_cache.exists():
            print(f"Tokenizer cache: {tokenizer_cache}")
            snapshots = list(tokenizer_cache.glob("snapshots/*/"))
            if snapshots:
                print(f"Snapshot directory: {snapshots[0]}")
                print(f"Contents: {os.listdir(snapshots[0])}")
            else:
                print("No snapshots found")
        
        print("\nTokenizer is now cached for offline use.")
        print("You can now run training in offline mode.")
        
    except Exception as e:
        print(f"Error downloading tokenizer: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download and cache a tokenizer for offline use")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Name of the pre-trained model to download tokenizer for")
    parser.add_argument("--cache_dir", type=str, default=str(get_cache_dir()),
                        help="Directory to save cached files")
    
    args = parser.parse_args()
    download_tokenizer(args.model_name, args.cache_dir)

if __name__ == "__main__":
    main() 