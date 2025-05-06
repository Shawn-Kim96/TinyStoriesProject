import torch
from torch.utils.data import Dataset
import re


class TinyStoriesBPEInfillingDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, min_story_length=5):
        """
        Enhanced dataset for infilling a story given the first and last sentence using a BPE tokenizer.
        
        Args:
            dataset: The HuggingFace dataset containing stories
            tokenizer: The BPE tokenizer wrapper
            max_length: Maximum length of sequences
            min_story_length: Minimum length of a story to be considered (in tokens)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_story_length = min_story_length
        
        self.processed_examples = []
        self._process_dataset()
    
    def _process_dataset(self):
        """Process the dataset using BPE tokenizer"""
        for item in self.dataset:
            text = item['text']
            
            # Split into sentences
            sentences = text.split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            # Skip stories that are too short
            if len(sentences) < 3:
                continue
                
            first_sentence = sentences[0]
            last_sentence = sentences[-1]
            
            # Create input: first sentence + <blank> + last sentence
            input_text = first_sentence + " <blank> " + last_sentence
            
            # Tokenize with BPE tokenizer
            input_encoding = self.tokenizer.encode_batch(
                [input_text], 
                padding=False, 
                truncation=True, 
                max_length=self.max_length
            )
            
            target_encoding = self.tokenizer.encode_batch(
                [text], 
                padding=False, 
                truncation=True, 
                max_length=self.max_length
            )
            
            # Get the token IDs
            input_ids = input_encoding["input_ids"][0].tolist()
            target_ids = target_encoding["input_ids"][0].tolist()
            
            # Skip if the input or target is too short
            if len(input_ids) < self.min_story_length or len(target_ids) < self.min_story_length:
                continue
            
            self.processed_examples.append({
                'input_ids': input_ids,
                'target_ids': target_ids,
                'first_sentence': first_sentence,
                'last_sentence': last_sentence,
                'full_story': text
            })
    
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        example = self.processed_examples[idx]
        
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'target_ids': torch.tensor(example['target_ids'], dtype=torch.long),
            'first_sentence': example['first_sentence'],
            'last_sentence': example['last_sentence'],
            'full_story': example['full_story']
        }
