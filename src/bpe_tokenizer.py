from transformers import AutoTokenizer
import torch
from pathlib import Path

class BPETokenizerWrapper:
    """
    A wrapper for Hugging Face's pre-trained BPE tokenizer to use in the TinyStories infilling model.
    This class provides a consistent interface for the rest of the project while using a pre-trained tokenizer.
    """
    def __init__(self, model_name="gpt2", special_tokens=None, cache_dir=None):
        """
        Initialize the BPE tokenizer wrapper.
        
        Args:
            model_name: The name of the pre-trained model to load tokenizer from
            special_tokens: A dictionary of special tokens to add to the tokenizer
            cache_dir: Directory to cache downloaded tokenizer files
        """
        # Set up cache directory for tokenizer
        if cache_dir:
            tokenizer_cache = Path(cache_dir) / "tokenizer"
            tokenizer_cache.mkdir(parents=True, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(tokenizer_cache))
            print(f"Using tokenizer cache directory: {tokenizer_cache}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        if special_tokens is None:
            special_tokens = {"blank_token": "<blank>"}
        
        # Add the blank token and any other special tokens
        special_tokens_dict = {}
        if "blank_token" in special_tokens and special_tokens["blank_token"] not in self.tokenizer.get_vocab():
            special_tokens_dict["additional_special_tokens"] = [special_tokens["blank_token"]]

        if special_tokens_dict:
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added} special tokens to the tokenizer")
        
        # Get important token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.pad_token_id
            
        self.unk_token_id = self.tokenizer.unk_token_id
        self.blank_token_id = self.tokenizer.convert_tokens_to_ids(special_tokens.get("blank_token", "<blank>"))
        
        # Cache vocab
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.idx2word = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text):
        """
        Encode a string into token IDs.
        
        Args:
            text: The text to encode
            
        Returns:
            A list of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def encode_batch(self, texts, padding=True, truncation=True, max_length=512):
        """
        Encode a batch of strings into token IDs.
        
        Args:
            texts: A list of strings to encode
            padding: Whether to pad sequences to the same length
            truncation: Whether to truncate sequences that exceed max_length
            max_length: The maximum length of encoded sequences
            
        Returns:
            A dictionary with "input_ids" containing tensors of token IDs
        """
        batch_encoding = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt"
        )
        return batch_encoding
    
    def decode(self, token_ids):
        """
        Decode token IDs into a string.
        
        Args:
            token_ids: A list or tensor of token IDs
            
        Returns:
            The decoded string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def get_vocab_size(self):
        """
        Get the size of the tokenizer's vocabulary.
        
        Returns:
            The vocabulary size
        """
        return self.vocab_size
    
    def get_word2idx(self):
        """
        Get the word to index mapping.
        
        Returns:
            A dictionary mapping tokens to indices
        """
        return self.vocab
    
    def get_idx2word(self):
        """
        Get the index to word mapping.
        
        Returns:
            A dictionary mapping indices to tokens
        """
        return self.idx2word 