from transformers import AutoTokenizer, AutoConfig
import torch
from pathlib import Path
import os
import json

class BPETokenizerWrapper:
    """
    A wrapper for Hugging Face's pre-trained BPE tokenizer to use in the TinyStories infilling model.
    This class provides a consistent interface for the rest of the project while using a pre-trained tokenizer.
    """
    def __init__(self, model_name="gpt2", special_tokens=None, cache_dir=None, offline_mode=False):
        """
        Initialize the BPE tokenizer wrapper.
        
        Args:
            model_name: The name of the pre-trained model to load tokenizer from
            special_tokens: A dictionary of special tokens to add to the tokenizer
            cache_dir: Directory to cache downloaded tokenizer files
            offline_mode: Whether to run in offline mode (no internet connection)
        """
        self.model_name = model_name
        self.special_tokens = special_tokens or {}
        self.offline_mode = offline_mode
        
        if cache_dir:
            # Configure paths according to Hugging Face cache structure
            base_cache = Path(cache_dir) / "tokenizers" / model_name
            tokenizer_cache = base_cache / f"models--{model_name.replace('/', '--')}"
            
            # Check for cached files in multiple possible locations
            possible_snapshot_dirs = list(tokenizer_cache.glob("snapshots/*/"))
            
            print(f"Using tokenizer cache directory: {tokenizer_cache}")
            
            if offline_mode:
                if possible_snapshot_dirs:
                    snapshot_dir = possible_snapshot_dirs[0]
                    print(f"Found snapshot directory: {snapshot_dir}")
                    # Directly specify the directory containing tokenizer files
                    model_path = str(snapshot_dir)
                else:
                    print(f"Warning: Could not find snapshot directory under {tokenizer_cache}")
                    # Try to use the tokenizer cache directly
                    model_path = str(tokenizer_cache)
            else:
                model_path = model_name
        else:
            base_cache = None
            model_path = model_name
        
        try:
            # Load the tokenizer, respecting offline mode setting
            print(f"Attempting to load tokenizer from {model_path} with offline_mode={offline_mode}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=str(base_cache) if base_cache else None,
                local_files_only=offline_mode,  # Only use local files in offline mode
                trust_remote_code=True
            )
            print("Successfully loaded tokenizer" + (" from local cache" if offline_mode else ""))
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            if cache_dir:
                print(f"Cache directory structure:")
                print(f"- Base cache: {base_cache}")
                print(f"- Tokenizer cache: {tokenizer_cache}")
                print(f"- Snapshots: {os.listdir(tokenizer_cache / 'snapshots') if (tokenizer_cache / 'snapshots').exists() else 'Not found'}")
                if possible_snapshot_dirs:
                    print(f"- Snapshot contents: {os.listdir(possible_snapshot_dirs[0])}")
            raise
        
        # Make sure we have necessary tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token to eos_token")

        # Make sure all special tokens are properly added
        blank_token = special_tokens.get("blank_token", "<blank>") if special_tokens else "<blank>"
        
        # 1. First check if blank token exists in vocabulary
        blank_token_id = self.tokenizer.convert_tokens_to_ids(blank_token) 
        
        # 2. If blank token is unknown token, add it explicitly
        if blank_token_id == self.tokenizer.unk_token_id:
            print(f"Adding {blank_token} as a special token to tokenizer")
            special_tokens_dict = {"additional_special_tokens": [blank_token]}
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added} special tokens to the tokenizer")
            
            # Also add it directly as a new token in vocabulary
            num_added = self.tokenizer.add_tokens([blank_token])
            print(f"Added {blank_token} directly to vocabulary, new tokens: {num_added}")
            
            # Get the new ID
            blank_token_id = self.tokenizer.convert_tokens_to_ids(blank_token)
            print(f"Blank token '{blank_token}' has ID: {blank_token_id}")
            
            # Sanity check
            if blank_token_id == self.tokenizer.unk_token_id:
                print(f"WARNING: Failed to add {blank_token} properly. Will use unk_token as fallback.")
        else:
            print(f"Blank token '{blank_token}' already in vocabulary with ID: {blank_token_id}")
        
        # Store special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.blank_token_id = blank_token_id
        self.bos_token_id = getattr(self.tokenizer, 'bos_token_id', None) or self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Print token IDs for debugging
        print(f"Token IDs - PAD: {self.pad_token_id}, BLANK: {self.blank_token_id}, BOS: {self.bos_token_id}, EOS: {self.eos_token_id}")
        
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
        return self.tokenizer.encode(text, add_special_tokens=True)
    
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
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
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