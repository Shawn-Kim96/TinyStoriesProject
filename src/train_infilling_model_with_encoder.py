import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
import argparse
import random
import numpy as np
from pathlib import Path
import pickle
import logging
import sys
from datetime import datetime

# Import the encoder-decoder classes
from src.dataset_with_encoder import TinyStoriesEncoderDecoderDataset, encoder_decoder_padding_collate_fn
from src.models_with_encoder import StoryInfillingEncoderDecoder
from src.bpe_tokenizer import BPETokenizerWrapper
from src.config import get_model_dir, get_data_dir, get_cache_dir, get_default_tokenizer_model


class TinyStoriesDataset:
    def __init__(self, data_dir, split, max_samples=None):
        self.data = []
        
        data_file = Path(data_dir) / f"{split}.jsonl"
        print(f"Loading data from {data_file}")
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples is not None and i >= max_samples:
                        break
                    self.data.append(json.loads(line))
            
            total_samples = len(self.data)
            max_info = f" (limited to {max_samples})" if max_samples is not None else ""
            print(f"Successfully loaded {total_samples} examples from {split} split{max_info}")
        except Exception as e:
            print(f"Error reading file {data_file}: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class OnlineTinyStoriesDataset:
    def __init__(self, split, max_samples=None):
        try:
            from datasets import load_dataset
            dataset = load_dataset("roneneldan/TinyStories", split=split)
            
            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            self.data = dataset
            total_samples = len(self.data)
            max_info = f" (limited to {max_samples})" if max_samples is not None else ""
            print(f"Successfully loaded {total_samples} examples from {split} split{max_info} using HuggingFace Datasets")
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def get_processed_dataset_cache_path(cache_dir, split, max_length, min_story_length, offline_mode, tokenizer_model, encoder_decoder_mode, max_samples=None):
    """Create a unique cache path for processed dataset based on parameters"""
    cache_dir = Path(cache_dir) / "processed_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename based on parameters that affect dataset processing
    mode = "offline" if offline_mode else "online"
    samples_info = f"_samples{max_samples}" if max_samples is not None else ""
    filename = f"{split}_encoder_decoder_{mode}_{max_length}_{min_story_length}{samples_info}_{os.path.basename(tokenizer_model)}_{encoder_decoder_mode}.pkl"
    return cache_dir / filename


def save_processed_dataset(dataset, cache_path):
    """Save processed dataset to cache"""
    print(f"Saving processed dataset to {cache_path}")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Successfully saved processed dataset to cache")
        return True
    except Exception as e:
        print(f"Warning: Failed to save processed dataset to cache: {str(e)}")
        return False


def load_processed_dataset(cache_path):
    """Load processed dataset from cache if available"""
    if not cache_path.exists():
        print(f"No cached dataset found at {cache_path}")
        return None
    
    print(f"Loading processed dataset from cache: {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Successfully loaded processed dataset from cache with {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"Warning: Failed to load processed dataset from cache: {str(e)}")
        return None


def setup_logging(model_dir):
    """Set up logging configuration"""
    log_file = model_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def train(args):
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using model directory: {model_dir}")
    
    # Set cache directory for datasets
    cache_dir = Path(args.cache_dir) / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    # Check if running in offline or online mode
    offline_mode = args.offline_mode
    print(f"Running in {'offline' if offline_mode else 'online'} mode")
    
    # Load dataset based on mode
    print("Loading dataset...")
    if offline_mode:
        # Load from local files in offline mode
        data_dir = Path(args.data_dir) / "tinystories_data"
        train_dataset = TinyStoriesDataset(data_dir, "train", args.max_samples)
        valid_dataset = TinyStoriesDataset(data_dir, "validation", args.max_samples)
    else:
        # Load directly from Hugging Face in online mode
        train_dataset = OnlineTinyStoriesDataset("train", args.max_samples)
        valid_dataset = OnlineTinyStoriesDataset("validation", args.max_samples)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    
    # Initialize BPE tokenizer
    print(f"Initializing BPE tokenizer from {args.tokenizer_model}...")
    special_tokens = {
        "blank_token": "<blank>"
    }
    tokenizer = BPETokenizerWrapper(
        model_name=args.tokenizer_model,
        special_tokens=special_tokens,
        cache_dir=args.cache_dir,
        offline_mode=offline_mode  # Pass the offline mode parameter
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocabulary size: {vocab_size}")
    print(f"Using blank_token_id: {tokenizer.blank_token_id}")
    
    # Verify blank token is properly added
    if tokenizer.blank_token_id == tokenizer.tokenizer.unk_token_id:
        print("WARNING: blank_token_id is the same as unk_token_id. This may cause issues during generation.")
    
    # Check for cached processed datasets
    train_cache_path = get_processed_dataset_cache_path(
        args.cache_dir, 
        "train", 
        args.max_seq_length, 
        args.min_story_length, 
        offline_mode, 
        args.tokenizer_model,
        args.max_samples,
        'encoder_decoder_true'
    )
    
    valid_cache_path = get_processed_dataset_cache_path(
        args.cache_dir, 
        "validation", 
        args.max_seq_length, 
        args.min_story_length, 
        offline_mode, 
        args.tokenizer_model,
        args.max_samples,
        'encoder_decoder_true'
    )
    
    # Try to load train data from cache
    train_data = load_processed_dataset(train_cache_path)
    if train_data is None:
        print("Creating train dataset...")
        train_data = TinyStoriesEncoderDecoderDataset(
            train_dataset, 
            tokenizer, 
            max_length=args.max_seq_length,
            min_story_length=args.min_story_length
        )
        # Save processed dataset to cache
        save_processed_dataset(train_data, train_cache_path)
    
    # Try to load validation data from cache
    valid_data = load_processed_dataset(valid_cache_path)
    if valid_data is None:
        print("Creating validation dataset...")
        valid_data = TinyStoriesEncoderDecoderDataset(
            valid_dataset, 
            tokenizer, 
            max_length=args.max_seq_length,
            min_story_length=args.min_story_length
        )
        # Save processed dataset to cache
        save_processed_dataset(valid_data, valid_cache_path)
    
    print(f"Processed train dataset size: {len(train_data)}")
    print(f"Processed validation dataset size: {len(valid_data)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=encoder_decoder_padding_collate_fn, 
        num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=encoder_decoder_padding_collate_fn, 
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing encoder-decoder model...")
    model = StoryInfillingEncoderDecoder(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
        blank_token_id=tokenizer.blank_token_id if tokenizer.blank_token_id != tokenizer.tokenizer.unk_token_id else None,
        bos_token_id=tokenizer.tokenizer.bos_token_id,
        eos_token_id=tokenizer.tokenizer.eos_token_id
    ).to(device)
    model_path = Path(model_dir) / f"tinystories_encoder_decoder_infilling_model_emb{args.embed_dim}_encoderlayer{args.num_encoder_layers}_decoderlayer{args.num_decoder_layers}_head{args.num_heads}_bs{args.batch_size}_seq{args.max_seq_length}.pth"
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Setup logging
    logger = setup_logging(model_dir)
    logger.info(f"Starting training with configuration: {args}")
    
    # Training loop
    print("Starting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to device
            encoder_input_ids = batch['encoder_input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_target_ids = batch['decoder_target_ids'].to(device)
            encoder_padding_mask = batch['encoder_padding_mask'].to(device) if 'encoder_padding_mask' in batch else None
            decoder_padding_mask = batch['decoder_padding_mask'].to(device) if 'decoder_padding_mask' in batch else None
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(encoder_input_ids, decoder_input_ids)
            
            # Calculate loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            target_flat = decoder_target_ids.reshape(-1)
            batch_loss = criterion(logits_flat, target_flat)
            
            # Backward and optimize
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            train_loss += batch_loss.item()
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {batch_loss.item():.4f}, Time: {time.time() - start_time:.2f}s")
                start_time = time.time()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                # Move tensors to device
                encoder_input_ids = batch['encoder_input_ids'].to(device)
                decoder_input_ids = batch['decoder_input_ids'].to(device)
                decoder_target_ids = batch['decoder_target_ids'].to(device)
                encoder_padding_mask = batch['encoder_padding_mask'].to(device) if 'encoder_padding_mask' in batch else None
                decoder_padding_mask = batch['decoder_padding_mask'].to(device) if 'decoder_padding_mask' in batch else None
                
                # Forward pass
                logits = model(encoder_input_ids, decoder_input_ids)
                
                # Calculate loss
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                target_flat = decoder_target_ids.reshape(-1)
                batch_loss = criterion(logits_flat, target_flat)
                
                valid_loss += batch_loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        
        # Generate a sample
        if (epoch + 1) % args.sample_interval == 0:
            model.eval()
            sample_idx = random.randint(0, len(valid_data) - 1)
            sample = valid_data[sample_idx]
            first_sentence = sample['first_sentence']
            last_sentence = sample['last_sentence']
            
            print("\nSample generation:")
            print(f"First sentence: {first_sentence}")
            print(f"Last sentence: {last_sentence}")
            
            # Generate story infilling using the model
            generated_story = model.generate(
                first_sentence=first_sentence, 
                last_sentence=last_sentence, 
                tokenizer=tokenizer, 
                max_length=100,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
                top_p=args.sample_top_p
            )
            
            print(f"Generated story: {generated_story}")
            print(f"Original story: {sample['full_story']}")
        
        # Save the model if it has the best validation loss so far
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            
            # Create model directory if it doesn't exist
            model_path = Path(model_dir) / 'tinystories_encoder_decoder_infilling_model.pth'
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
                'tokenizer_model': args.tokenizer_model,
                'args': vars(args)
            }, model_path)
            print(f"Model saved to {model_path}")
        
        logger.info(f"Epoch {epoch + 1} completed. Loss: {avg_valid_loss:.4f}")
    
    logger.info("Training completed!")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a TinyStories infilling model with Encoder-Decoder architecture')
    
    # File paths
    parser.add_argument('--data_dir', type=str, default=get_data_dir(), 
                        help='Directory with data files')
    parser.add_argument('--model_dir', type=str, default=get_model_dir(), 
                        help='Directory to save model checkpoints')
    parser.add_argument('--cache_dir', type=str, default=get_cache_dir(), 
                        help='Directory to cache downloaded files')
                        
    # Dataset parameters
    parser.add_argument('--offline_mode', action='store_true', 
                        help='Run in offline mode with local files')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to use (for debugging)')
    parser.add_argument('--max_seq_length', type=int, default=128, 
                        help='Maximum sequence length')
    parser.add_argument('--min_story_length', type=int, default=5, 
                        help='Minimum story length in tokens')
    parser.add_argument('--tokenizer_model', type=str, default=get_default_tokenizer_model(), 
                        help='Tokenizer model to use')
    
    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=384, 
                        help='Embedding dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=3, 
                        help='Number of transformer layers in encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=3, 
                        help='Number of transformer layers in decoder')
    parser.add_argument('--num_heads', type=int, default=6, 
                        help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=1536, 
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                        help='Gradient clipping')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loader workers')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, 
                        help='Teacher forcing ratio (used during training)')
    
    # Sampling parameters
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Generate sample every N epochs')
    parser.add_argument('--sample_temperature', type=float, default=0.8,
                        help='Temperature for sampling during generation')
    parser.add_argument('--sample_top_k', type=int, default=50,
                        help='Top-k sampling parameter during generation')
    parser.add_argument('--sample_top_p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling parameter during generation')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, 
                        help='Log interval in batch iterations')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)