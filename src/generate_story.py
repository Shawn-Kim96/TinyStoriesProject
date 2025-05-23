import torch
import argparse
import os
from pathlib import Path
from src.models import StoryInfillingModel
from src.bpe_tokenizer import BPETokenizerWrapper
from src.config import get_model_dir, get_cache_dir


def generate_story(
    first_sentence,
    last_sentence,
    model_path=None,
    max_tokens=200,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    device=None,
    cache_dir=None
):
    """
    Generate a story with the specified first and last sentences using a pre-trained model.
    
    Args:
        first_sentence: The first sentence of the story.
        last_sentence: The last sentence of the story.
        model_path: Path to the trained model checkpoint.
        max_tokens: Maximum number of tokens to generate.
        temperature: Temperature for sampling (higher = more random).
        top_k: Top-k sampling parameter (0 to disable).
        top_p: Top-p (nucleus) sampling parameter (1.0 to disable).
        device: Device to use for inference (None for auto-detection).
        cache_dir: Directory to use for caching downloaded models.
        
    Returns:
        The generated story.
    """
    # Use default model path if not specified
    if model_path is None:
        model_dir = get_model_dir()
        model_path = Path(model_dir) / "tinystories_bpe_infilling_model.pth"
    
    # Use default cache directory if not specified
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['args']
    
    # Initialize tokenizer
    tokenizer_model = checkpoint.get('tokenizer_model', 'gpt2')
    print(f"Loading tokenizer from {tokenizer_model}...")
    
    tokenizer = BPETokenizerWrapper(
        model_name=tokenizer_model,
        special_tokens={"blank_token": "<blank>"},
        cache_dir=cache_dir
    )
    
    # Initialize model
    model = StoryInfillingModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=model_args['embed_dim'],
        num_layers=model_args['num_layers'],
        num_heads=model_args['num_heads'],
        ff_dim=model_args['ff_dim'],
        max_seq_length=model_args['max_seq_length'],
        dropout=model_args['dropout'],
        pad_token_id=tokenizer.pad_token_id,
        blank_token_id=tokenizer.blank_token_id
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Generating story...")
    
    # Generate the story
    generated_story = model.generate(
        first_sentence=first_sentence,
        last_sentence=last_sentence,
        tokenizer=tokenizer,
        max_length=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        teacher_forcing_ratio=0.0  # We don't use teacher forcing during generation
    )
    
    return generated_story


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a story given first and last sentences")
    
    parser.add_argument('--first_sentence', type=str, required=True, help='The first sentence of the story')
    parser.add_argument('--last_sentence', type=str, required=True, help='The last sentence of the story')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the saved model (default: use model_dir from config)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for cache files (default: use cache_dir from config)')
    parser.add_argument('--max_tokens', type=int, default=200, 
                        help='Maximum number of tokens to generate for the middle part')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature for sampling (higher = more creative)')
    parser.add_argument('--top_k', type=int, default=50, 
                        help='Number of top tokens to consider for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, 
                        help='Cumulative probability for nucleus sampling')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use for inference (cuda or cpu)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    story = generate_story(
        first_sentence=args.first_sentence,
        last_sentence=args.last_sentence,
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    print("\nGenerated Story:")
    print("-" * 50)
    print(story)
    print("-" * 50) 