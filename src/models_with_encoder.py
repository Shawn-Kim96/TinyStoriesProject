import torch
import torch.nn as nn


# EncoderBlock
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention (bidirectional in encoder)
        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            is_causal=False  # No causal masking in encoder
        )
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# DecoderBlock
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Self-attention
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, causal_mask=None, key_padding_mask=None, cross_key_padding_mask=None):
        # Self-attention
        self_attn_output, _ = self.self_attention(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            is_causal=True  # Causal masking in decoder self-attention
        )
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention (attending to encoder outputs)
        cross_attn_output, _ = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=cross_key_padding_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_length, dropout=0.1, pad_token_id=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device
        
        # Create padding mask
        key_padding_mask = (x == self.pad_token_id) if self.pad_token_id is not None else None
        
        # Embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        x = self.dropout(token_embed + pos_embed)
        
        # Process through encoder blocks
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            
        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_length, dropout=0.1, pad_token_id=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def generate_causal_mask(self, seq_len, device):
        # Standard lower triangular mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, encoder_output, encoder_padding_mask=None):
        batch_size, seq_len = x.size()
        device = x.device
        
        # Create padding masks
        key_padding_mask = (x == self.pad_token_id) if self.pad_token_id is not None else None
        
        # Create causal attention mask for self-attention
        causal_mask = self.generate_causal_mask(seq_len, device)
        
        # Embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        x = self.dropout(token_embed + pos_embed)
        
        # Process through decoder blocks
        for layer in self.layers:
            x = layer(
                x, 
                encoder_output, 
                causal_mask=causal_mask, 
                key_padding_mask=key_padding_mask,
                cross_key_padding_mask=encoder_padding_mask
            )
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        return logits


# Complete Encoder-Decoder model
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_encoder_layers, num_decoder_layers, 
                 num_heads, ff_dim, max_seq_length, dropout=0.1, pad_token_id=None):
        super().__init__()
        self.encoder = Encoder(
            vocab_size, embed_dim, num_encoder_layers, num_heads, 
            ff_dim, max_seq_length, dropout, pad_token_id
        )
        self.decoder = Decoder(
            vocab_size, embed_dim, num_decoder_layers, num_heads, 
            ff_dim, max_seq_length, dropout, pad_token_id
        )
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        
    def forward(self, encoder_input, decoder_input):
        # Get encoder padding mask for cross attention
        encoder_padding_mask = (encoder_input == self.pad_token_id) if self.pad_token_id is not None else None
        
        # Process through encoder
        encoder_output = self.encoder(encoder_input)
        
        # Process through decoder with cross-attention to encoder output
        logits = self.decoder(decoder_input, encoder_output, encoder_padding_mask)
        
        return logits


class StoryInfillingEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_encoder_layers=3, num_decoder_layers=3, 
                 num_heads=8, ff_dim=2048, max_seq_length=512, dropout=0.1, 
                 pad_token_id=0, blank_token_id=None, bos_token_id=None, eos_token_id=None):
        """
        An encoder-decoder model that fills in the middle part of a story given the first and last sentences.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            num_encoder_layers: Number of transformer layers in encoder
            num_decoder_layers: Number of transformer layers in decoder
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            pad_token_id: ID for padding token
            blank_token_id: ID for blank token that marks where to infill
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
        """
        super().__init__()
        self.transformer = EncoderDecoderTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_seq_length = max_seq_length
        
    def forward(self, encoder_input, decoder_input=None):
        """
        Forward pass for training.
        
        Args:
            encoder_input: Input to encoder (first + last sentences) of shape (batch_size, seq_len)
            decoder_input: Input to decoder of shape (batch_size, seq_len)
        
        Returns:
            logits: Output logits of shape (batch_size, decoder_seq_len, vocab_size)
        """
        return self.transformer(encoder_input, decoder_input)
    
    def _prepare_encoder_input(self, first_sentence, last_sentence, tokenizer):
        """
        Prepare encoder input by concatenating first and last sentences.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: The last sentence of the story
            tokenizer: The tokenizer for encoding
            
        Returns:
            encoder_input: Tensor of token IDs for the encoder
        """
        # For encoder input, we concatenate first and last sentence
        # We can add special tokens to help the model distinguish them
        input_text = first_sentence + " <blank> " + last_sentence
        input_encoding = tokenizer.encode_batch([input_text], padding=False, truncation=True, max_length=self.max_seq_length)
        encoder_input = torch.tensor(input_encoding["input_ids"][0], device=self._get_device())
        
        return encoder_input.unsqueeze(0)  # Add batch dimension
    
    def _prepare_decoder_input(self, batch_size=1, start_token_id=None):
        """
        Prepare initial decoder input (usually just the BOS token).
        
        Args:
            batch_size: Batch size
            start_token_id: Token ID to start generation with
            
        Returns:
            decoder_input: Initial decoder input tensor
        """
        device = self._get_device()
        
        if start_token_id is None:
            start_token_id = self.bos_token_id if self.bos_token_id is not None else 0
            
        # Start with just the BOS token
        decoder_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        return decoder_input
    
    def _get_device(self):
        """Get the device the model is on"""
        return next(self.parameters()).device
    
    def generate(self, first_sentence, last_sentence=None, tokenizer=None, max_length=200, 
                temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate text by infilling between first and last sentences.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: The last sentence (required for encoder-decoder model)
            tokenizer: The tokenizer for encoding/decoding
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_k: Number of top tokens to consider for sampling
            top_p: Cumulative probability for nucleus sampling
        
        Returns:
            The generated story (with filled middle part)
        """
        if last_sentence is None:
            # For encoder-decoder model, we need both sentences
            raise ValueError("last_sentence is required for the encoder-decoder model")
            
        self.eval()
        device = self._get_device()
        
        # Prepare encoder input
        encoder_input = self._prepare_encoder_input(first_sentence, last_sentence, tokenizer)
        
        # Run encoder once (this is efficient compared to decoder-only approach)
        with torch.no_grad():
            encoder_output = self.transformer.encoder(encoder_input)
            encoder_padding_mask = (encoder_input == self.pad_token_id) if self.pad_token_id is not None else None
        
        # Start with BOS token for decoder input
        decoder_input = self._prepare_decoder_input()
        
        # Autoregressive generation
        generated_tokens = []
        
        for _ in range(max_length):
            with torch.no_grad():
                # Forward through decoder only
                logits = self.transformer.decoder(decoder_input, encoder_output, encoder_padding_mask)
                
                # Get next-token logits (last position)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
 
                    # Shift the indices to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')

                # Sample from filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Check for NaN or negative values in probs
                if torch.isnan(probs).any() or (probs < 0).any():
                    # Fix problematic probabilities
                    probs = torch.nan_to_num(probs, nan=1e-7, posinf=1e-7, neginf=1e-7)
                    # Ensure non-negative
                    probs = torch.clamp(probs, min=1e-7)
                    # Renormalize
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Safety check for sum
                if not (0.99 < probs.sum() < 1.01):
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Use softmax again to be safe
                try:
                    next_token = torch.multinomial(probs, num_samples=1).item()
                except Exception as e:
                    # Fallback: just pick most likely token
                    print(f"Sampling failed, falling back to argmax: {e}")
                    next_token = torch.argmax(probs).item()
                
                # Stop if EOS token is generated
                if next_token == self.eos_token_id:
                    break
                    
                # Add to generated tokens
                generated_tokens.append(next_token)
                
                # Extend decoder input for next iteration
                decoder_input = torch.cat([
                    decoder_input, 
                    torch.tensor([[next_token]], device=device)
                ], dim=1)
        
        # Create the complete story
        middle_part = tokenizer.decode(generated_tokens)
        
        # Replace the blank token with the generated middle
        result = first_sentence + " " + middle_part + " " + last_sentence
        
        # Clean up the text 
        result = result.replace("<blank>", "").replace("  ", " ")
        
        return result
    
    def train_forward(self, first_sentence, last_sentence, target_text, tokenizer):
        """
        Training forward pass for teacher forcing.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: The last sentence of the story
            target_text: The text to generate (middle part)
            tokenizer: The tokenizer for encoding/decoding
            
        Returns:
            tuple: (logits, target) for loss calculation
        """
        device = self._get_device()
        
        # Prepare encoder input (first + last sentences)
        encoder_input = self._prepare_encoder_input(first_sentence, last_sentence, tokenizer)
        
        # For decoder input, we use the target text shifted right
        target_encoding = tokenizer.encode_batch([target_text], padding=False, truncation=True, max_length=self.max_seq_length)
        target_ids = torch.tensor(target_encoding["input_ids"][0], device=device)
        
        # Create decoder input (target shifted right with BOS at start)
        if self.bos_token_id is not None:
            decoder_input = torch.cat([
                torch.tensor([self.bos_token_id], device=device),
                target_ids[:-1]  # Remove last token
            ])
        else:
            # If no BOS token, just shift right with padding
            decoder_input = torch.cat([
                torch.tensor([self.pad_token_id], device=device),
                target_ids[:-1]  # Remove last token
            ])
        
        # Add batch dimension
        decoder_input = decoder_input.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)
        
        # Forward pass through encoder-decoder
        logits = self(encoder_input, decoder_input)
        
        return logits, target_ids
    
    def train_step(self, first_sentence, last_sentence, target_text, tokenizer, criterion):
        """
        Perform a training step with teacher forcing.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: The last sentence of the story
            target_text: The target text to generate
            tokenizer: The tokenizer for encoding/decoding
            criterion: Loss function
            
        Returns:
            loss: Training loss
        """
        # Get logits and target
        logits, target = self.train_forward(first_sentence, last_sentence, target_text, tokenizer)
        
        # Calculate loss
        # Reshape logits and target for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)
        
        loss = criterion(logits_flat, target_flat)
        
        return loss