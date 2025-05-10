import torch
from torch.utils.data import Dataset
import re


class TinyStoriesEncoderDecoderDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, min_story_length=5):
        """
        Dataset for encoder-decoder story infilling using TinyStories.
        
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
        """Process the dataset for encoder-decoder architecture"""
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
            
            # Extract middle part (target for generation)
            middle_part = ' '.join(sentences[1:-1])
            
            # Create encoder input: first sentence + <blank> + last sentence
            encoder_input_text = first_sentence + " <blank> " + last_sentence
            
            # Decoder input/target is just the middle part
            decoder_target_text = middle_part
            
            # Tokenize with BPE tokenizer
            encoder_encoding = self.tokenizer.encode_batch(
                [encoder_input_text], 
                padding=False, 
                truncation=True, 
                max_length=self.max_length
            )
            
            decoder_encoding = self.tokenizer.encode_batch(
                [decoder_target_text], 
                padding=False, 
                truncation=True, 
                max_length=self.max_length
            )
            
            # Get the token IDs
            encoder_input_ids = encoder_encoding["input_ids"][0].tolist()
            decoder_target_ids = decoder_encoding["input_ids"][0].tolist()
            
            # Skip if input or target is too short
            if len(encoder_input_ids) < self.min_story_length or len(decoder_target_ids) < self.min_story_length:
                continue
            
            # For decoder input (teacher forcing), shift right and add BOS token
            bos_token_id = getattr(self.tokenizer, 'bos_token_id', 0)  # Default to 0 if not available
            decoder_input_ids = [bos_token_id] + decoder_target_ids[:-1]  # Add BOS, remove last token
            
            self.processed_examples.append({
                'encoder_input_ids': encoder_input_ids,
                'decoder_input_ids': decoder_input_ids,
                'decoder_target_ids': decoder_target_ids,
                'first_sentence': first_sentence,
                'last_sentence': last_sentence,
                'middle_part': middle_part,
                'full_story': text
            })
    
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        example = self.processed_examples[idx]
        
        return {
            'encoder_input_ids': torch.tensor(example['encoder_input_ids'], dtype=torch.long),
            'decoder_input_ids': torch.tensor(example['decoder_input_ids'], dtype=torch.long),
            'decoder_target_ids': torch.tensor(example['decoder_target_ids'], dtype=torch.long),
            'first_sentence': example['first_sentence'],
            'last_sentence': example['last_sentence'],
            'middle_part': example['middle_part'],
            'full_story': example['full_story']
        }


def encoder_decoder_padding_collate_fn(batch):
    """
    Collate function that pads sequences in a batch for encoder-decoder models.
    """
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    decoder_target_ids = [item['decoder_target_ids'] for item in batch]
    
    # Get other fields
    first_sentences = [item['first_sentence'] for item in batch]
    last_sentences = [item['last_sentence'] for item in batch]
    middle_parts = [item['middle_part'] for item in batch]
    full_stories = [item['full_story'] for item in batch]
    
    # Find max length in the batch
    max_encoder_len = max(len(ids) for ids in encoder_input_ids)
    max_decoder_len = max(len(ids) for ids in decoder_input_ids)
    max_target_len = max(len(ids) for ids in decoder_target_ids)
    
    # Pad sequences
    padded_encoder_inputs = []
    padded_decoder_inputs = []
    padded_targets = []
    encoder_padding_masks = []
    decoder_padding_masks = []
    
    for enc_inp, dec_inp, dec_tgt in zip(encoder_input_ids, decoder_input_ids, decoder_target_ids):
        # Pad encoder input
        enc_inp_padding = torch.zeros(max_encoder_len - len(enc_inp), dtype=torch.long)
        padded_enc_inp = torch.cat([torch.tensor(enc_inp, dtype=torch.long), enc_inp_padding])
        padded_encoder_inputs.append(padded_enc_inp)
        
        # Create encoder padding mask (1 for real tokens, 0 for padding)
        enc_padding_mask = torch.ones(len(enc_inp), dtype=torch.bool)
        enc_mask_padding = torch.zeros(max_encoder_len - len(enc_inp), dtype=torch.bool)
        enc_padding_mask = torch.cat([enc_padding_mask, enc_mask_padding])
        encoder_padding_masks.append(enc_padding_mask)
        
        # Pad decoder input
        dec_inp_padding = torch.zeros(max_decoder_len - len(dec_inp), dtype=torch.long)
        padded_dec_inp = torch.cat([torch.tensor(dec_inp, dtype=torch.long), dec_inp_padding])
        padded_decoder_inputs.append(padded_dec_inp)
        
        # Create decoder padding mask
        dec_padding_mask = torch.ones(len(dec_inp), dtype=torch.bool)
        dec_mask_padding = torch.zeros(max_decoder_len - len(dec_inp), dtype=torch.bool)
        dec_padding_mask = torch.cat([dec_padding_mask, dec_mask_padding])
        decoder_padding_masks.append(dec_padding_mask)
        
        # Pad target
        tgt_padding = torch.zeros(max_target_len - len(dec_tgt), dtype=torch.long)
        padded_tgt = torch.cat([torch.tensor(dec_tgt, dtype=torch.long), tgt_padding])
        padded_targets.append(padded_tgt)
    
    return {
        'encoder_input_ids': torch.stack(padded_encoder_inputs),
        'decoder_input_ids': torch.stack(padded_decoder_inputs),
        'decoder_target_ids': torch.stack(padded_targets),
        'encoder_padding_mask': torch.stack(encoder_padding_masks),
        'decoder_padding_mask': torch.stack(decoder_padding_masks),
        'first_sentences': first_sentences,
        'last_sentences': last_sentences,
        'middle_parts': middle_parts,
        'full_stories': full_stories
    }