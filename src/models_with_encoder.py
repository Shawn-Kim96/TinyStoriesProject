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
            nn.GELU(),  # ReLU 대신 GELU 사용 (더 부드러운 활성화 함수)
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 스케일링 팩터
        self.attn_scale = 0.125
        self.ff_scale = 0.125

    def forward(self, x, encoder_output, causal_mask=None, key_padding_mask=None, cross_key_padding_mask=None):
        # 입력 텐서 복사 (안전을 위해)
        residual = x.clone()
        
        try:
            # Self-attention with safety checks
            self_attn_output, _ = self.self_attention(
                x, x, x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                is_causal=True  # Causal masking in decoder self-attention
            )
            
            # NaN 체크 및 처리
            if torch.isnan(self_attn_output).any() or torch.isinf(self_attn_output).any():
                print("Warning: NaN or Inf detected in self-attention output, applying fix")
                self_attn_output = torch.nan_to_num(self_attn_output, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 스케일링 적용하여 그래디언트 안정화
            self_attn_output = self_attn_output * self.attn_scale
            
            # 잔차 연결 및 정규화
            x = self.norm1(residual + self.dropout(self_attn_output))
        except Exception as e:
            print(f"Error in self-attention: {e}")
            # 오류 발생 시 입력 유지
            x = self.norm1(residual)
        
        # 크로스 어텐션 연산을 위한 잔차 저장
        residual = x.clone()
        
        try:
            # Cross-attention (attending to encoder outputs) with safety checks
            cross_attn_output, _ = self.cross_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                key_padding_mask=cross_key_padding_mask
            )
            
            # NaN 체크 및 처리
            if torch.isnan(cross_attn_output).any() or torch.isinf(cross_attn_output).any():
                print("Warning: NaN or Inf detected in cross-attention output, applying fix")
                cross_attn_output = torch.nan_to_num(cross_attn_output, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 스케일링 적용하여 그래디언트 안정화
            cross_attn_output = cross_attn_output * self.attn_scale
            
            # 잔차 연결 및 정규화
            x = self.norm2(residual + self.dropout(cross_attn_output))
        except Exception as e:
            print(f"Error in cross-attention: {e}")
            # 오류 발생 시 입력 유지
            x = self.norm2(residual)
        
        # 피드포워드 연산을 위한 잔차 저장
        residual = x.clone()
        
        try:
            # Feed forward with safety checks
            ff_output = self.feed_forward(x)
            
            # NaN 체크 및 처리
            if torch.isnan(ff_output).any() or torch.isinf(ff_output).any():
                print("Warning: NaN or Inf detected in feed-forward output, applying fix")
                ff_output = torch.nan_to_num(ff_output, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 스케일링 적용하여 그래디언트 안정화
            ff_output = ff_output * self.ff_scale
            
            # 잔차 연결 및 정규화
            x = self.norm3(residual + self.dropout(ff_output))
        except Exception as e:
            print(f"Error in feed-forward: {e}")
            # 오류 발생 시 입력 유지
            x = self.norm3(residual)
        
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
        
        # 출력 레이어를 두 단계로 나누어 안정성 향상
        self.pre_output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        # 가중치 초기화 스케일 감소
        with torch.no_grad():
            self.fc_out.weight.data.mul_(0.1)
            
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
        for i, layer in enumerate(self.layers):
            try:
                x = layer(
                    x, 
                    encoder_output, 
                    causal_mask=causal_mask, 
                    key_padding_mask=key_padding_mask,
                    cross_key_padding_mask=encoder_padding_mask
                )
                # 중간 레이어에서 NaN 체크 및 처리
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning: NaN or Inf detected in decoder layer {i} output, applying fix")
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e:
                print(f"Error in decoder layer {i}: {e}")
                # 오류 발생 시 이전 값 유지
                continue
        
        # 출력 전 추가 처리 레이어
        x = self.pre_output(x)
        
        # Project to vocabulary with 그래디언트 스케일링
        logits = self.fc_out(x)
        
        # 출력 로짓의 스케일 조정 (너무 큰 값 방지)
        logits = logits / 1.5
        
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
        
        # 가중치 초기화를 통한 안정적인 학습
        self._init_weights()
    
    def _init_weights(self):
        """모델 가중치를 적절하게 초기화"""
        # 임베딩 레이어 초기화
        if hasattr(self.transformer.encoder, 'token_embedding'):
            nn.init.normal_(self.transformer.encoder.token_embedding.weight, mean=0.0, std=0.02)
        if hasattr(self.transformer.decoder, 'token_embedding'):
            nn.init.normal_(self.transformer.decoder.token_embedding.weight, mean=0.0, std=0.02)
            
        # 포지션 임베딩 초기화
        if hasattr(self.transformer.encoder, 'position_embedding'):
            nn.init.normal_(self.transformer.encoder.position_embedding.weight, mean=0.0, std=0.02)
        if hasattr(self.transformer.decoder, 'position_embedding'):
            nn.init.normal_(self.transformer.decoder.position_embedding.weight, mean=0.0, std=0.02)
        
        # 선형 레이어 초기화
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 선형 레이어 가중치 초기화
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # 레이어 정규화 초기화
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
        
        print(f"Generating with temperature={temperature}, top_k={top_k}, top_p={top_p}")
        
        # Prepare encoder input
        encoder_input = self._prepare_encoder_input(first_sentence, last_sentence, tokenizer)
        print(f"Encoder input shape: {encoder_input.shape}")
        
        try:
            # Run encoder once (this is efficient compared to decoder-only approach)
            with torch.no_grad():
                encoder_output = self.transformer.encoder(encoder_input)
                print(f"Encoder output shape: {encoder_output.shape}")
                encoder_padding_mask = (encoder_input == self.pad_token_id) if self.pad_token_id is not None else None
            
            # Start with BOS token for decoder input
            decoder_input = self._prepare_decoder_input()
            print(f"Initial decoder input shape: {decoder_input.shape}")
            
            # Autoregressive generation
            generated_tokens = []
            
            for i in range(max_length):
                with torch.no_grad():
                    # Forward through decoder only
                    logits = self.transformer.decoder(decoder_input, encoder_output, encoder_padding_mask)
                    
                    # Get next-token logits (last position)
                    next_token_logits = logits[:, -1, :]
                    
                    # 로짓 값 안정화
                    if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                        print(f"Warning: NaN or Inf detected in generation logits at step {i}")
                        next_token_logits = torch.zeros_like(next_token_logits)
                        # 랜덤한 토큰 선택으로 대체
                        next_token = torch.randint(0, self.transformer.decoder.fc_out.out_features, (1,)).item()
                        generated_tokens.append(next_token)
                        decoder_input = torch.cat([
                            decoder_input, 
                            torch.tensor([[next_token]], device=device)
                        ], dim=1)
                        continue
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / max(0.1, temperature)
                    
                    # 극단적인 값 클리핑
                    next_token_logits = torch.clamp(next_token_logits, min=-50.0, max=50.0)
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_values)

                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        
                        # Shift the indices to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Set indices to remove to -inf
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[0, indices_to_remove] = float('-inf')

                    # Sample from filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # Ensure we have valid probabilities
                    if torch.isnan(probs).any() or (probs < 0).any():
                        print(f"Warning: Found NaN or negative probabilities at step {i}")
                        probs = torch.nan_to_num(probs, nan=1e-8)
                        probs = torch.clamp(probs, min=1e-8)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    try:
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    except Exception as e:
                        print(f"Sampling failed at step {i}, falling back to argmax: {e}")
                        next_token = torch.argmax(next_token_logits).item()
                    
                    # Stop if EOS token is generated or we've reached max length
                    if next_token == self.eos_token_id or next_token == tokenizer.tokenizer.eos_token_id:
                        print(f"Generated EOS token at step {i}")
                        break
                        
                    # Skip blank token and unknown token
                    if next_token == tokenizer.blank_token_id or next_token == tokenizer.tokenizer.unk_token_id:
                        print(f"Skipping special token {next_token} at step {i}")
                        continue
                    
                    # Add to generated tokens
                    generated_tokens.append(next_token)
                    
                    # Extend decoder input for next iteration
                    decoder_input = torch.cat([
                        decoder_input, 
                        torch.tensor([[next_token]], device=device)
                    ], dim=1)
                    
                    # Print progress every 10 tokens
                    if i % 10 == 0:
                        print(f"Generated {i} tokens so far")
            
            # Create the complete story
            if not generated_tokens:
                print("Warning: No tokens were generated")
                middle_part = ""
            else:
                middle_part = tokenizer.decode(generated_tokens)
                print(f"Generated middle part: {middle_part}")
            
            # Replace the blank token with the generated middle
            result = first_sentence + " " + middle_part + " " + last_sentence
            
            # Clean up the text 
            result = result.replace("<blank>", "").replace("  ", " ")
            
            return result
            
        except Exception as e:
            print(f"Error in generation process: {e}")
            # Return a fallback response if generation fails
            result = first_sentence + " " + last_sentence
            return result.replace("<blank>", "").replace("  ", " ")
    
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