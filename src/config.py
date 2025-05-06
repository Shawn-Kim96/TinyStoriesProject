"""
Configuration module for the TinyStories project.
Handles environment variables and path configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Get data directory from environment variable or use default
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "model"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / "cache"))

# Specific dataset paths
TINYSTORIES_DATA = Path(os.getenv("TINYSTORIES_DATA", DATA_DIR / "tinystories"))
TOKENIZER_CACHE = Path(os.getenv("TOKENIZER_CACHE", CACHE_DIR / "tokenizer"))

# Default tokenizer model
DEFAULT_TOKENIZER_MODEL = os.getenv("DEFAULT_TOKENIZER_MODEL", "gpt2")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TINYSTORIES_DATA.mkdir(parents=True, exist_ok=True)
TOKENIZER_CACHE.mkdir(parents=True, exist_ok=True)

def get_data_dir():
    """Return the data directory path."""
    return DATA_DIR

def get_model_dir():
    """Return the model directory path."""
    return MODEL_DIR

def get_cache_dir():
    """Return the cache directory path."""
    return CACHE_DIR

def get_tinystories_data_dir():
    """Return the TinyStories dataset directory path."""
    return TINYSTORIES_DATA

def get_tokenizer_cache_dir():
    """Return the tokenizer cache directory path."""
    return TOKENIZER_CACHE

def get_default_tokenizer_model():
    """Return the default tokenizer model name."""
    return DEFAULT_TOKENIZER_MODEL 