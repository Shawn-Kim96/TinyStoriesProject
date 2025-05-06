import torch
import torch.nn as nn


# TransformerBlock
class TransformerBlock(nn.Module):
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

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            is_causal=True
        )
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Generate infiling attention mask
def create_infilling_attention_mask(x, blank_token_id):
    batch_size, seq_len = x.size()
    device = x.device
    attn_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)

    for batch_index, input_seq in enumerate(x):
        try:
            blank_idx = (input_seq == blank_token_id).nonzero(as_tuple=True)[0].item()
        except IndexError:
            blank_idx = seq_len

        # causal mask (make attention mask -inf for words after blank)
        for i in range(blank_idx + 1, seq_len):
            for j in range(i+1, seq_len):
                attn_mask[batch_index, i, j] = float('-inf')

    return attn_mask


# DecoderOnlyTransformer
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_length, dropout=0.1, pad_token_id=None, blank_token_id=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        self.max_seq_length = max_seq_length

    def generate_causal_mask(self, seq_len, device):
        # GPT-style causal mask (standard lower triangular mask)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device

        # --- Generate attention mask ---
        attn_mask = self.generate_causal_mask(seq_len, device)  # <=== causal mask를 만든다

        key_padding_mask = (x == self.pad_token_id) if self.pad_token_id is not None else None

        # --- Embedding ---
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        x = self.dropout(token_embed + pos_embed)

        # --- Transformer Blocks 통과 ---
        for layer in self.layers:
            x = layer(x, causal_mask=attn_mask, key_padding_mask=key_padding_mask)

        # --- Output ---
        logits = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        return logits


class StoryInfillingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, num_heads=8, ff_dim=2048, 
                 max_seq_length=512, dropout=0.1, pad_token_id=0, blank_token_id=None):
        """
        A model that generates the middle part of a story given the first and last sentences.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            pad_token_id: ID for padding token
            blank_token_id: ID for blank token that marks where to infill
        """
        super().__init__()
        self.transformer = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_token_id=pad_token_id,
            blank_token_id=blank_token_id
        )
        
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        self.max_seq_length = max_seq_length
        
    def forward(self, x):
        """
        Forward pass for training.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token IDs
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        return self.transformer(x)
    
    def _extract_middle_from_ground_truth(self, first_sentence, last_sentence, ground_truth, tokenizer):
        """
        Helper function to extract the middle part from ground truth.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: The last sentence of the story
            ground_truth: The complete ground truth story
            tokenizer: The tokenizer for encoding/decoding
            
        Returns:
            middle_ids: Tensor of token IDs representing the middle part
        """
        device = next(self.parameters()).device
        
        # Encode sentences
        first_encoding = tokenizer.encode_batch([first_sentence], padding=False, truncation=True)
        last_encoding = tokenizer.encode_batch([last_sentence], padding=False, truncation=True)
        ground_truth_encoding = tokenizer.encode_batch([ground_truth], padding=False, truncation=True, max_length=self.max_seq_length)
        
        first_ids = first_encoding["input_ids"][0]
        last_ids = last_encoding["input_ids"][0]
        gt_ids = ground_truth_encoding["input_ids"][0].to(device)
        
        first_length = len(first_ids)
        last_length = len(last_ids)
        
        # Simple heuristic to extract middle: 
        # remove first part tokens from the beginning and last part tokens from the end
        middle_start = first_length
        middle_end = len(gt_ids) - last_length
        
        if middle_end <= middle_start:
            print("Warning: Could not extract middle part from ground truth. Using full ground truth minus first sentence.")
            middle_ids = gt_ids[first_length:]
        else:
            middle_ids = gt_ids[middle_start:middle_end]
            
        return middle_ids
    
    def _find_context_match_position(self, context_tokens, reference_tokens, step, fallback_position):
        """
        Helper function to find matching context in the reference tokens.
        
        Args:
            context_tokens: Context tokens to match
            reference_tokens: Reference tokens to search in
            step: Current generation step
            fallback_position: Fallback position if no match is found
            
        Returns:
            position: Position in reference tokens after the matching context
        """
        device = next(self.parameters()).device
        context_len = len(context_tokens)
        
        if context_len == 0:
            return fallback_position
            
        # Find where context appears in reference
        position = -1
        for i in range(len(reference_tokens) - context_len):
            if torch.equal(reference_tokens[i:i+context_len], context_tokens):
                position = i + context_len + step
                break
                
        # Fallback if no match or position is out of bounds
        if position == -1 or position >= len(reference_tokens):
            position = min(fallback_position, len(reference_tokens) - 1)
            
        return position
    
    def generate_or_train(self, first_sentence, last_sentence=None, tokenizer=None, max_tokens=200,
                          temperature=1.0, top_k=50, top_p=0.9, teacher_forcing_ratio=1.0, 
                          ground_truth=None, is_training=False):
        """
        Unified function for generation and training with teacher forcing.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: Optional last sentence of the story (if None, will generate continuation)
            tokenizer: The tokenizer for encoding/decoding
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_k: Number of top tokens to consider for sampling
            top_p: Cumulative probability for nucleus sampling
            teacher_forcing_ratio: Probability of using ground truth token (0.0-1.0)
            ground_truth: Optional ground truth story for teacher forcing
            is_training: Whether this is a training run (affects gradient calculation)
            
        Returns:
            If training: Tuple of (generated_text, loss)
            If not training: Generated text
        """
        if is_training:
            self.train()
            if teacher_forcing_ratio < 1.0:
                print("Warning: Using teacher_forcing_ratio < 1.0 in training mode may slow down learning.")
        else:
            self.eval()
            
        device = next(self.parameters()).device
        
        # Check if ground_truth is provided when teacher_forcing_ratio > 0
        if teacher_forcing_ratio > 0 and ground_truth is None:
            print("Warning: teacher_forcing_ratio > 0 but no ground_truth provided. Will fall back to normal generation.")
            teacher_forcing_ratio = 0.0
            
        # Prepare input based on whether last_sentence is provided
        if last_sentence:
            # Infilling mode: first + <blank> + last
            input_text = first_sentence + " <blank> " + last_sentence
            input_encoding = tokenizer.encode_batch([input_text], padding=False, truncation=True, max_length=self.max_seq_length)
            input_ids = input_encoding["input_ids"][0].to(device)
            
            # Find blank token position
            try:
                blank_idx = (input_ids == self.blank_token_id).nonzero(as_tuple=True)[0].item()
            except IndexError:
                print("Warning: <blank> token not found in input. Using a default position.")
                blank_idx = len(input_ids) // 2
                
            use_infilling = True
        else:
            # Continuation mode: just use first sentence as input
            input_encoding = tokenizer.encode_batch([first_sentence], padding=False, truncation=True, max_length=self.max_seq_length)
            input_ids = input_encoding["input_ids"][0].to(device)  # encoding output = (batch_size, seq_len)
            blank_idx = len(input_ids) - 1  # Last token position
            use_infilling = False
            
        # Process ground truth for teacher forcing if provided
        ground_truth_ids = None
        middle_ids = None
        
        if ground_truth is not None and teacher_forcing_ratio > 0:
            gt_encoding = tokenizer.encode_batch([ground_truth], padding=False, truncation=True, max_length=self.max_seq_length)
            ground_truth_ids = gt_encoding["input_ids"][0].to(device)
            
            # Extract middle part if in infilling mode
            if use_infilling and last_sentence:
                middle_ids = self._extract_middle_from_ground_truth(
                    first_sentence, last_sentence, ground_truth, tokenizer
                )
                
        # Setup for training
        losses = [] if is_training else None
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id) if is_training else None
        current_input_ids = input_ids.clone()
        generated_tokens = []
        
        # Generation loop
        with torch.set_grad_enabled(is_training):
            for step in range(max_tokens):
                # If sequence is too long, truncate
                if len(current_input_ids) >= self.max_seq_length:
                    break
                    
                # Forward pass
                logits = self(current_input_ids.unsqueeze(0))
                
                # Get logits at the appropriate position (blank or last)
                if use_infilling:
                    next_token_logits = logits[0, blank_idx, :] / temperature
                else:
                    next_token_logits = logits[0, -1, :] / temperature
                
                # Calculate loss for training
                if is_training and middle_ids is not None and step < len(middle_ids):
                    # Get target token
                    target_token = middle_ids[step]
                    # Calculate loss
                    loss = criterion(next_token_logits.unsqueeze(0), target_token.unsqueeze(0))
                    losses.append(loss)
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
 
                    # We are going to save the word where it started to exceed top_p cumulative p.
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    # if the first word had the highest prob exceeding top_p, we have to keep that word.
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from filtered distribution or use teacher forcing
                use_teacher_forcing = (teacher_forcing_ratio > 0 and 
                                      torch.rand(1).item() < teacher_forcing_ratio and
                                      (middle_ids is not None or ground_truth_ids is not None))
                
                if use_teacher_forcing:
                    if use_infilling and middle_ids is not None and step < len(middle_ids):
                        # Use token from extracted middle part
                        next_token = middle_ids[step].item()
                    elif ground_truth_ids is not None:
                        # For continuation or if middle extraction failed, find position in full ground truth
                        if use_infilling:
                            # Try to find context match in ground truth
                            context_len = min(5, blank_idx)
                            context_start = max(0, blank_idx - context_len)
                            context_tokens = current_input_ids[context_start:blank_idx]
                            
                            gt_position = self._find_context_match_position(
                                context_tokens, ground_truth_ids, step, blank_idx + step + 1
                            )
                        else:
                            # For continuation, use position after first sentence
                            gt_position = min(len(current_input_ids) + step, len(ground_truth_ids) - 1)
                            
                        next_token = ground_truth_ids[gt_position].item()
                    else:
                        # Fallback to sampling
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    # Use model's prediction
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if EOS token or pad token is generated
                if next_token == tokenizer.tokenizer.eos_token_id or next_token == self.pad_token_id:
                    break
                
                # Add to generated tokens
                generated_tokens.append(next_token)
                
                # Update input sequence for next iteration
                if use_infilling:
                    # Replace blank with generated token and add a new blank
                    new_input_ids = torch.cat([
                        current_input_ids[:blank_idx],
                        torch.tensor([next_token], device=device),
                        torch.tensor([self.blank_token_id], device=device),
                        current_input_ids[blank_idx+1:]
                    ])
                    
                    current_input_ids = new_input_ids
                    
                    # Ensure we don't exceed the maximum sequence length
                    if len(current_input_ids) > self.max_seq_length:
                        current_input_ids = current_input_ids[:self.max_seq_length]
                    
                    # Update blank index
                    try:
                        blank_idx = (current_input_ids == self.blank_token_id).nonzero(as_tuple=True)[0].item()
                    except IndexError:
                        # If no blank token, we're done
                        break
                else:
                    # For continuation, just append the generated token
                    current_input_ids = torch.cat([current_input_ids, torch.tensor([next_token], device=device)])
        
        # Decode the generated sequence
        if use_infilling:
            generated_text = tokenizer.decode(current_input_ids.tolist())
            # Clean up the text by removing the blank token marker
            generated_text = generated_text.replace("<blank>", "")
        else:
            generated_text = tokenizer.decode(current_input_ids.tolist())
        
        # Return based on mode
        if is_training and losses:
            avg_loss = torch.stack(losses).mean()
            return generated_text, avg_loss
        else:
            return generated_text
    
    def generate(self, first_sentence, last_sentence=None, tokenizer=None, max_length=200, 
                temperature=1.0, top_k=50, top_p=0.9, teacher_forcing_ratio=1.0, ground_truth=None):
        """
        Generate text, either by infilling between first and last sentences or continuing from first sentence.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: Optional last sentence (if None, will generate continuation)
            tokenizer: The tokenizer for encoding/decoding
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_k: Number of top tokens to consider for sampling
            top_p: Cumulative probability for nucleus sampling
            teacher_forcing_ratio: Probability of using ground truth token (0.0-1.0)
            ground_truth: Optional ground truth story for teacher forcing
        
        Returns:
            The generated story
        """
        return self.generate_or_train(
            first_sentence=first_sentence,
            last_sentence=last_sentence,
            tokenizer=tokenizer,
            max_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            teacher_forcing_ratio=teacher_forcing_ratio,
            ground_truth=ground_truth,
            is_training=False
        )
    
    def train_with_teacher_forcing(self, first_sentence, last_sentence=None, ground_truth=None, 
                                  tokenizer=None, max_tokens=200, teacher_forcing_ratio=1.0):
        """
        Train the model with teacher forcing, either for infilling or continuation.
        
        Args:
            first_sentence: The first sentence of the story
            last_sentence: Optional last sentence (if None, will train continuation)
            ground_truth: The complete ground truth story
            tokenizer: The tokenizer for encoding/decoding
            max_tokens: Maximum tokens to generate or train
            teacher_forcing_ratio: Probability of using ground truth token (0.0-1.0)
        
        Returns:
            A tuple of (generated_story, loss)
        """
        if ground_truth is None:
            raise ValueError("ground_truth must be provided for training with teacher forcing")
            
        return self.generate_or_train(
            first_sentence=first_sentence,
            last_sentence=last_sentence,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            temperature=1.0,  # Fixed temperature for training
            top_k=50,         # Fixed top_k for training
            top_p=0.9,        # Fixed top_p for training
            teacher_forcing_ratio=teacher_forcing_ratio,
            ground_truth=ground_truth,
            is_training=True
        )