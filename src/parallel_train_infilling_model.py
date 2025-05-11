import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import json
import time
import os
import argparse
import random
import numpy as np
from pathlib import Path
from glob import glob
import pickle
import logging
import sys
from datetime import datetime

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


def setup_logging(model_dir, rank):
    """Set up logging configuration"""
    # Only log to file from rank 0
    if rank == 0:
        log_file = model_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        # For other ranks, only log to console with rank prefix
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    return logging.getLogger(__name__)


def setup_distributed(rank, world_size):
    """
    Initialize distributed training environment
    """
    # Set the distributed initialization method to env:// backend
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    

def cleanup_distributed():
    """
    Clean up distributed training environment
    """
    dist.destroy_process_group()


def train_distributed(rank, world_size, args):
    # Initialize distributed environment
    setup_distributed(rank, world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Set random seeds for reproducibility
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # Setup logging
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(model_dir, rank)
    
    if rank == 0:
        logger.info(f"Training with {world_size} GPUs")
        logger.info(f"Using model directory: {model_dir}")
    
    # Set cache directory for datasets
    cache_dir = Path(args.cache_dir) / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if running in offline or online mode
    offline_mode = args.offline_mode
    
    # Load dataset based on mode (only rank 0 prints progress)
    if rank == 0:
        logger.info("Loading dataset...")

    if offline_mode:
        # Load from local files in offline mode
        data_dir = Path(args.data_dir) / "tinystories_data"
        train_dataset = TinyStoriesDataset(data_dir, "train", args.max_samples)
        valid_dataset = TinyStoriesDataset(data_dir, "validation", args.max_samples)
    else:
        # Load directly from Hugging Face in online mode
        train_dataset = OnlineTinyStoriesDataset("train", args.max_samples)
        valid_dataset = OnlineTinyStoriesDataset("validation", args.max_samples)
    
    if rank == 0:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(valid_dataset)}")
    
    # Initialize BPE tokenizer
    if rank == 0:
        logger.info(f"Initializing BPE tokenizer from {args.tokenizer_model}...")
        
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
    
    if rank == 0:
        logger.info(f"Tokenizer vocabulary size: {vocab_size}")
    
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
    
    # Try to load train data from cache - only rank 0 needs to do this if it's not cached yet
    if rank == 0:
        train_data = load_processed_dataset(train_cache_path)
        if train_data is None:
            logger.info("Creating train dataset...")
            train_data = TinyStoriesBPEInfillingDataset(
                train_dataset, 
                tokenizer, 
                max_length=args.max_seq_length,
                min_story_length=args.min_story_length
            )
            # Save processed dataset to cache
            save_processed_dataset(train_data, train_cache_path)
            
        valid_data = load_processed_dataset(valid_cache_path)
        if valid_data is None:
            logger.info("Creating validation dataset...")
            valid_data = TinyStoriesBPEInfillingDataset(
                valid_dataset, 
                tokenizer, 
                max_length=args.max_seq_length,
                min_story_length=args.min_story_length
            )
            # Save processed dataset to cache
            save_processed_dataset(valid_data, valid_cache_path)
    
    # Make sure all processes wait for rank 0 to finish creating and caching datasets
    dist.barrier()
    
    # Now all ranks can load from cache
    if rank != 0:
        train_data = load_processed_dataset(train_cache_path)
        valid_data = load_processed_dataset(valid_cache_path)
    
    if rank == 0:
        logger.info(f"Processed train dataset size: {len(train_data)}")
        logger.info(f"Processed validation dataset size: {len(valid_data)}")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    valid_sampler = DistributedSampler(
        valid_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed
    )
    
    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        collate_fn=padding_collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_data, 
        batch_size=args.batch_size, 
        sampler=valid_sampler,
        collate_fn=padding_collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    if rank == 0:
        logger.info("Initializing model...")
        
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
    
    # Wrap model in DistributedDataParallel
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if rank == 0:
        logger.info(f"Starting training with configuration: {args}")
    
    # Training loop
    best_valid_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
            
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device) if 'target_ids' in batch else None
            padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
            
            # Determine whether to use vectorized batch processing or individual processing
            optimizer.zero_grad()
            
            # Process the entire batch at once for better performance
            batch_logits = model(input_ids)
            
            # Shift the target sequence for next-token prediction
            shift_logits = batch_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            batch_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward and optimize
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            train_loss += batch_loss.item()
            
            if (batch_idx + 1) % args.log_interval == 0 and rank == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch [{epoch+1}/{args.num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {batch_loss.item():.4f}, "
                    f"Speed: {args.batch_size * world_size * args.log_interval / elapsed:.2f} samples/sec"
                )
                start_time = time.time()
        
        # Calculate average loss across all processes
        train_loss = train_loss / len(train_loader)
        avg_train_loss = torch.tensor([train_loss], device=device)
        dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = avg_train_loss.item() / world_size
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                
                # Process the entire batch at once for better performance
                batch_logits = model(input_ids)
                
                # Shift the target sequence for next-token prediction
                shift_logits = batch_logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                batch_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                valid_loss += batch_loss.item()
        
        # Calculate average validation loss across all processes
        valid_loss = valid_loss / len(valid_loader)
        avg_valid_loss = torch.tensor([valid_loss], device=device)
        dist.all_reduce(avg_valid_loss, op=dist.ReduceOp.SUM)
        avg_valid_loss = avg_valid_loss.item() / world_size
        
        if rank == 0:
            logger.info(
                f"Epoch [{epoch+1}/{args.num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Valid Loss: {avg_valid_loss:.4f}"
            )
        
        # Only rank 0 generates samples and saves model
        if rank == 0:
            # Generate a sample
            if (epoch + 1) % args.sample_interval == 0:
                # Get a model in eval mode that's not wrapped in DDP for generation
                sample_model = model.module
                
                sample_idx = random.randint(0, len(valid_data) - 1)
                sample = valid_data[sample_idx]
                first_sentence = sample['first_sentence']
                last_sentence = sample['last_sentence']
                
                logger.info("\nSample generation:")
                logger.info(f"First sentence: {first_sentence}")
                logger.info(f"Last sentence: {last_sentence}")
                
                # Generate story using generate
                generated_story = sample_model.generate(
                    first_sentence=first_sentence, 
                    last_sentence=last_sentence, 
                    tokenizer=tokenizer, 
                    max_length=100,
                    teacher_forcing_ratio=0.0,  # During testing, we don't want teacher forcing
                )
                
                logger.info(f"Generated story: {generated_story}")
                logger.info(f"Original story: {sample['full_story']}")
            
            # Save the model if it has the best validation loss so far
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                
                # Create model directory if it doesn't exist
                model_path = Path(model_dir) / 'tinystories_bpe_infilling_model.pth'
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),  # Save the unwrapped model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'valid_loss': avg_valid_loss,
                    'tokenizer_model': args.tokenizer_model,
                    'args': vars(args)
                }, model_path)
                logger.info(f"Model saved to {model_path}")
        
        # Wait for all processes to finish the epoch
        dist.barrier()
    
    if rank == 0:
        logger.info("Training completed!")
    
    # Clean up distributed environment
    cleanup_distributed()


def train(args):
    """Wrapper function that launches the distributed training"""
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        print("Warning: Only one GPU detected. Falling back to single GPU training.")
        args.distributed = False
        # Simulate rank 0 training
        train_distributed(0, 1, args)
    else:
        args.distributed = True
        print(f"Starting distributed training with {world_size} GPUs")
        # Launch multiple processes for distributed training
        mp.spawn(
            train_distributed,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


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
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training (will be auto-detected)')
    
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
                        help='Batch size per GPU')
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