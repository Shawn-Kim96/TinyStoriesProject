import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import time
import os
import argparse
import random
import numpy as np
from pathlib import Path
from glob import glob
import pickle

from src.dataset import TinyStoriesBPEInfillingDataset
from src.models import StoryInfillingModel
from src.bpe_tokenizer import BPETokenizerWrapper
from src.config import get_model_dir, get_data_dir, get_cache_dir, get_default_tokenizer_model


class TinyStoriesDataset(Dataset):
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


class OnlineTinyStoriesDataset(Dataset):
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


def padding_collate_fn(batch):
    """
    Collate function that pads sequences in a batch to the same length.
    """
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    # Get other fields
    first_sentences = [item['first_sentence'] for item in batch]
    last_sentences = [item['last_sentence'] for item in batch]
    full_stories = [item['full_story'] for item in batch]
    
    # Find max length in the batch
    max_input_len = max(len(ids) for ids in input_ids)
    max_target_len = max(len(ids) for ids in target_ids)
    
    # Pad sequences
    padded_inputs = []
    padded_targets = []
    padding_masks = []
    
    for inp, tgt in zip(input_ids, target_ids):
        # Pad input
        inp_padding = torch.zeros(max_input_len - len(inp), dtype=torch.long)
        padded_inp = torch.cat([inp, inp_padding])
        padded_inputs.append(padded_inp)
        
        # Create padding mask (1 for real tokens, 0 for padding)
        padding_mask = torch.ones(len(inp), dtype=torch.long)
        mask_padding = torch.zeros(max_input_len - len(inp), dtype=torch.long)
        padding_mask = torch.cat([padding_mask, mask_padding])
        padding_masks.append(padding_mask)
        
        # Pad target
        tgt_padding = torch.zeros(max_target_len - len(tgt), dtype=torch.long)
        padded_tgt = torch.cat([tgt, tgt_padding])
        padded_targets.append(padded_tgt)
    
    return {
        'input_ids': torch.stack(padded_inputs),
        'target_ids': torch.stack(padded_targets),
        'padding_mask': torch.stack(padding_masks),
        'first_sentences': first_sentences,
        'last_sentences': last_sentences,
        'full_stories': full_stories
    }


def get_processed_dataset_cache_path(cache_dir, split, max_length, min_story_length, offline_mode, tokenizer_model, max_samples=None):
    """Create a unique cache path for processed dataset based on parameters"""
    cache_dir = Path(cache_dir) / "processed_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename based on parameters that affect dataset processing
    mode = "offline" if offline_mode else "online"
    samples_info = f"_samples{max_samples}" if max_samples is not None else ""
    filename = f"{split}_{mode}_{max_length}_{min_story_length}{samples_info}_{os.path.basename(tokenizer_model)}.pkl"
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


def process_batch_vectorized(model, batch, tokenizer, device, teacher_forcing_ratio, max_tokens, criterion):
    """
    Process a batch of data in a vectorized way for improved performance.
    This function is optimized to process the entire batch at once rather than one item at a time.
    """
    input_ids = batch['input_ids'].to(device)
    padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
    target_ids = batch['target_ids'].to(device) if 'target_ids' in batch else None
    
    # Forward pass
    logits = model(input_ids)  # shape: [batch_size, seq_len, vocab_size]
    
    # Calculate loss
    if target_ids is not None:
        # Reshape logits and target_ids for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        target_ids_flat = target_ids.view(-1)
        
        loss = criterion(logits_flat, target_ids_flat)
        return loss
    else:
        return None


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
    
    # Check for cached processed datasets
    train_cache_path = get_processed_dataset_cache_path(
        args.cache_dir, 
        "train", 
        args.max_seq_length, 
        args.min_story_length, 
        offline_mode, 
        args.tokenizer_model,
        args.max_samples
    )
    
    valid_cache_path = get_processed_dataset_cache_path(
        args.cache_dir, 
        "validation", 
        args.max_seq_length, 
        args.min_story_length, 
        offline_mode, 
        args.tokenizer_model,
        args.max_samples
    )
    
    # Try to load train data from cache
    train_data = load_processed_dataset(train_cache_path)
    if train_data is None:
        print("Creating train dataset...")
        train_data = TinyStoriesBPEInfillingDataset(
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
        valid_data = TinyStoriesBPEInfillingDataset(
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
        collate_fn=padding_collate_fn, 
        num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=padding_collate_fn, 
        num_workers=args.num_workers
    )
    
    # Initialize model
    # Check if model already exists
    model = StoryInfillingModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            max_seq_length=args.max_seq_length,
            dropout=args.dropout,
            pad_token_id=tokenizer.pad_token_id,
            blank_token_id=tokenizer.blank_token_id
    ).to(device)
    model_path = Path(model_dir) / f"tinystories_bpe_infilling_model_emb{args.embed_dim}_layer{args.num_layers}_head{args.num_heads}_bs{args.batch_size}_seq{args.max_seq_length}.pth"
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device) if 'target_ids' in batch else None
            padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
            first_sentences = batch['first_sentences']
            last_sentences = batch['last_sentences']
            full_stories = batch['full_stories']
            
            # Determine whether to use vectorized batch processing or individual processing
            use_vectorized = False
            if target_ids is not None and all(isinstance(s, str) for s in full_stories):
                use_vectorized = True
            
            optimizer.zero_grad()
            
            if use_vectorized:
                # Process the entire batch at once for better performance
                batch_loss = model(input_ids)
                
                # Shift the target sequence for next-token prediction
                shift_logits = batch_loss[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                batch_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                # Fallback to per-example processing when vectorized approach can't be used
                batch_loss = 0.0
                for i in range(len(first_sentences)):
                    # Use train_with_teacher_forcing to get loss
                    result = model.train_with_teacher_forcing(
                        first_sentence=first_sentences[i],
                        last_sentence=last_sentences[i],
                        ground_truth=full_stories[i],
                        tokenizer=tokenizer,
                        max_tokens=args.max_seq_length,
                        teacher_forcing_ratio=args.teacher_forcing_ratio
                    )
                    
                    # Handle the result correctly
                    if isinstance(result, tuple):
                        _, loss = result
                    else:
                        loss = result
                    
                    batch_loss += loss
                
                # Only normalize if we processed multiple examples individually
                if len(first_sentences) > 1:
                    batch_loss = batch_loss / len(first_sentences)
            
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
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device) if 'target_ids' in batch else None
                padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
                first_sentences = batch['first_sentences']
                last_sentences = batch['last_sentences']
                full_stories = batch['full_stories']
                
                # Determine whether to use vectorized batch processing or individual processing
                use_vectorized = False
                if target_ids is not None and all(isinstance(s, str) for s in full_stories):
                    use_vectorized = True
                
                if use_vectorized:
                    # Process the entire batch at once for better performance
                    batch_logits = model(input_ids)
                    
                    # Shift the target sequence for next-token prediction
                    shift_logits = batch_logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    
                    # Compute loss
                    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                    batch_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    # Fallback to per-example processing
                    batch_loss = 0.0
                    
                    for i in range(len(first_sentences)):
                        # Use train_with_teacher_forcing to evaluate
                        result = model.train_with_teacher_forcing(
                            first_sentence=first_sentences[i],
                            last_sentence=last_sentences[i],
                            ground_truth=full_stories[i],
                            tokenizer=tokenizer,
                            max_tokens=args.max_seq_length,
                            teacher_forcing_ratio=0.0  # Use 0.0 for validation to test real generation capability
                        )
                        
                        # Handle the result correctly
                        if isinstance(result, tuple):
                            _, loss = result
                        else:
                            loss = result
                        
                        batch_loss += loss
                    
                    # Only normalize if we processed multiple examples individually
                    if len(first_sentences) > 1:
                        batch_loss = batch_loss / len(first_sentences)
                
                valid_loss += batch_loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        
        # Generate a sample
        if (epoch + 1) % args.sample_interval == 0:
            sample_idx = random.randint(0, len(valid_data) - 1)
            sample = valid_data[sample_idx]
            first_sentence = sample['first_sentence']
            last_sentence = sample['last_sentence']
            
            print("\nSample generation:")
            print(f"First sentence: {first_sentence}")
            print(f"Last sentence: {last_sentence}")
            
            # Generate story using generate
            generated_story = model.generate(
                first_sentence=first_sentence, 
                last_sentence=last_sentence, 
                tokenizer=tokenizer, 
                max_length=100,
                teacher_forcing_ratio=0.0,  # During testing, we don't want teacher forcing
            )
            
            print(f"Generated story: {generated_story}")
            print(f"Original story: {sample['full_story']}")
        
        # Save the model if it has the best validation loss so far
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
                        
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a story infilling model on TinyStories dataset with BPE tokenizer")
    
    # Path parameters
    parser.add_argument('--model_dir', type=str, default=str(get_model_dir()),
                        help='Directory to save model checkpoints')
    parser.add_argument('--data_dir', type=str, default=str(get_data_dir()),
                        help='Directory for datasets')
    parser.add_argument('--cache_dir', type=str, default=str(get_cache_dir()),
                        help='Directory for cache files')
    
    # Environment parameters
    parser.add_argument('--offline_mode', action='store_true',
                        help='Run in offline mode (no internet access)')
    
    # Dataset parameters
    parser.add_argument('--tokenizer_model', type=str, default=get_default_tokenizer_model(), 
                        help='Pre-trained model to use for BPE tokenizer')
    parser.add_argument('--max_seq_length', type=int, default=256, 
                        help='Maximum sequence length')
    parser.add_argument('--min_story_length', type=int, default=5, 
                        help='Minimum story length to include (in tokens)')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=384, 
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=6, 
                        help='Number of transformer layers')
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
                        help='Teacher forcing ratio')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, 
                        help='Log interval in batches')
    parser.add_argument('--sample_interval', type=int, default=1, 
                        help='Sample generation interval in epochs')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to load')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)