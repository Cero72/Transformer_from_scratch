import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from embedding_layer import PretrainedTransformerEmbedding
from attention import MultiHeadAttention, EncoderBlock, DecoderBlock

class Transformer(nn.Module):
    """
    Complete Transformer model for machine translation as described in 'Attention Is All You Need'.
    
    This implementation includes:
    - Separate embedding layers for source and target languages
    - Multi-layer encoder and decoder stacks
    - Support for pretrained embeddings
    - Final output projection to vocabulary size
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_src_seq_length: int = 5000,
        max_tgt_seq_length: int = 5000,
        src_embedding_type: str = 'random',
        tgt_embedding_type: str = 'random',
        share_embeddings: bool = False,
        freeze_embeddings: bool = False
    ):
        """
        Initialize the Transformer model.
        
        Args:
            src_vocab_size: Size of source language vocabulary
            tgt_vocab_size: Size of target language vocabulary
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Hidden dimension of the feed-forward networks
            dropout: Dropout probability
            max_src_seq_length: Maximum source sequence length
            max_tgt_seq_length: Maximum target sequence length
            src_embedding_type: Type of source embeddings ('random', 'glove', 'word2vec', or path to custom embeddings)
            tgt_embedding_type: Type of target embeddings ('random', 'glove', 'word2vec', or path to custom embeddings)
            share_embeddings: Whether to share embeddings between encoder and decoder (useful for same language)
            freeze_embeddings: Whether to freeze pretrained embeddings
        """
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Source and target embeddings
        self.src_embedding = PretrainedTransformerEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            max_seq_length=max_src_seq_length,
            embedding_type=src_embedding_type,
            dropout=dropout,
            freeze_embeddings=freeze_embeddings
        )
        
        # Share embeddings if specified (useful for same language tasks)
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = PretrainedTransformerEmbedding(
                vocab_size=tgt_vocab_size,
                d_model=d_model,
                max_seq_length=max_tgt_seq_length,
                embedding_type=tgt_embedding_type,
                dropout=dropout,
                freeze_embeddings=freeze_embeddings
            )
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer for output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize model parameters using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Create a mask to hide padding tokens.
        
        Args:
            seq: Input sequence tensor [batch_size, seq_length]
            pad_idx: Index of the padding token
            
        Returns:
            Mask tensor [batch_size, 1, 1, seq_length]
        """
        # Create mask: 1 for non-padding tokens, 0 for padding
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """
        Create a causal mask for the decoder to prevent attending to future tokens.
        
        Args:
            seq_length: Length of the sequence
            
        Returns:
            Causal mask tensor [1, 1, seq_length, seq_length]
        """
        # Create lower triangular matrix
        mask = torch.tril(torch.ones(1, 1, seq_length, seq_length))
        return mask
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the source sequence.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_length]
            src_mask: Optional mask for padding tokens [batch_size, 1, 1, src_seq_length]
            
        Returns:
            Encoder output tensor [batch_size, src_seq_length, d_model]
        """
        # Create padding mask if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        
        # Apply source embedding
        x = self.src_embedding(src)  # [batch_size, src_seq_length, d_model]
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        
        return x
    
    def decode(
        self, 
        tgt: torch.Tensor, 
        encoder_output: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode the target sequence.
        
        Args:
            tgt: Target sequence tensor [batch_size, tgt_seq_length]
            encoder_output: Output from the encoder [batch_size, src_seq_length, d_model]
            src_mask: Optional mask for padding in source [batch_size, 1, 1, src_seq_length]
            tgt_mask: Optional mask for padding in target [batch_size, 1, 1, tgt_seq_length]
            causal_mask: Optional causal mask for decoder [1, 1, tgt_seq_length, tgt_seq_length]
            
        Returns:
            Decoder output tensor [batch_size, tgt_seq_length, d_model]
        """
        # Create padding mask if not provided
        if tgt_mask is None:
            tgt_mask = self.create_padding_mask(tgt)
        
        # Create causal mask if not provided
        if causal_mask is None:
            tgt_seq_length = tgt.size(1)
            causal_mask = self.create_causal_mask(tgt_seq_length).to(tgt.device)
            
            # Combine padding mask and causal mask
            if tgt_mask is not None:
                # Convert to float for multiplication instead of bitwise AND
                combined_mask = tgt_mask.float() * causal_mask.float()
                tgt_mask = combined_mask
            else:
                tgt_mask = causal_mask
        
        # Apply target embedding
        x = self.tgt_embedding(tgt)  # [batch_size, tgt_seq_length, d_model]
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer model.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_length]
            tgt: Target sequence tensor [batch_size, tgt_seq_length]
            src_mask: Optional mask for padding in source [batch_size, 1, 1, src_seq_length]
            tgt_mask: Optional mask for padding in target [batch_size, 1, 1, tgt_seq_length]
            
        Returns:
            Output logits tensor [batch_size, tgt_seq_length, tgt_vocab_size]
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        
        # Encode the source sequence
        encoder_output = self.encode(src, src_mask)
        
        # Decode the target sequence
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = self.output_projection(decoder_output)
        
        return output
    
    def greedy_decode(
        self, 
        src: torch.Tensor, 
        start_symbol_idx: int, 
        end_symbol_idx: int, 
        max_length: int = 100
    ) -> torch.Tensor:
        """
        Perform greedy decoding for inference.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_length]
            start_symbol_idx: Index of the start symbol in target vocabulary
            end_symbol_idx: Index of the end symbol in target vocabulary
            max_length: Maximum length of the generated sequence
            
        Returns:
            Generated sequence tensor [batch_size, seq_length]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode the source sequence
        src_mask = self.create_padding_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize target sequence with start symbol
        ys = torch.ones(batch_size, 1).fill_(start_symbol_idx).long().to(device)
        
        # Generate tokens one by one
        for i in range(max_length - 1):
            # Create causal mask
            tgt_mask = self.create_causal_mask(ys.size(1)).to(device)
            
            # Decode current sequence
            out = self.decode(ys, encoder_output, src_mask, tgt_mask)
            
            # Project to vocabulary and get next token
            prob = self.output_projection(out[:, -1])
            next_word = torch.argmax(prob, dim=-1, keepdim=True)
            
            # Concatenate to output sequence
            ys = torch.cat([ys, next_word], dim=1)
            
            # Check if all sequences have end symbol
            if (next_word == end_symbol_idx).all():
                break
        
        return ys
    
    def beam_search_decode(
        self, 
        src: torch.Tensor, 
        start_symbol_idx: int, 
        end_symbol_idx: int, 
        beam_size: int = 5, 
        max_length: int = 100,
        length_penalty: float = 0.6
    ) -> List[torch.Tensor]:
        """
        Perform beam search decoding for better translation quality.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_length]
            start_symbol_idx: Index of the start symbol in target vocabulary
            end_symbol_idx: Index of the end symbol in target vocabulary
            beam_size: Beam size for search
            max_length: Maximum length of the generated sequence
            length_penalty: Penalty factor for sequence length
            
        Returns:
            List of generated sequences, one for each item in the batch
        """
        batch_size = src.size(0)
        device = src.device
        
        # We'll implement beam search for one sequence at a time for simplicity
        generated_sequences = []
        
        for batch_idx in range(batch_size):
            # Get single source sequence
            src_seq = src[batch_idx:batch_idx+1]
            
            # Encode the source sequence
            src_mask = self.create_padding_mask(src_seq)
            encoder_output = self.encode(src_seq, src_mask)
            
            # Initialize with start token
            curr_tokens = torch.ones(1, 1).fill_(start_symbol_idx).long().to(device)
            beam_scores = torch.zeros(1).to(device)
            
            # Lists to store finished sequences and their scores
            finished_sequences = []
            finished_scores = []
            
            # Generate tokens using beam search
            for step in range(max_length):
                num_candidates = curr_tokens.size(0)
                curr_length = curr_tokens.size(1)
                
                # Create causal mask
                tgt_mask = self.create_causal_mask(curr_length).to(device)
                
                # Expand encoder output for each beam candidate
                expanded_encoder_output = encoder_output.expand(num_candidates, -1, -1)
                expanded_src_mask = src_mask.expand(num_candidates, -1, -1, -1)
                
                # Decode current sequences
                out = self.decode(curr_tokens, expanded_encoder_output, expanded_src_mask, tgt_mask)
                
                # Project to vocabulary and get log probabilities
                logits = self.output_projection(out[:, -1])
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Calculate scores for next tokens
                vocab_size = log_probs.size(-1)
                next_scores = beam_scores.unsqueeze(1) + log_probs
                
                # Flatten scores to find top-k
                flat_next_scores = next_scores.view(-1)
                
                # Select top-k scores and their indices
                if step == 0:
                    # For first step, we only expand from the first beam
                    top_scores, top_indices = flat_next_scores[:vocab_size].topk(beam_size, dim=0)
                else:
                    top_scores, top_indices = flat_next_scores.topk(beam_size, dim=0)
                
                # Convert flat indices to beam indices and token indices
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # Create new beam candidates
                new_tokens = []
                new_scores = []
                
                # Process each beam candidate
                for beam_idx, token_idx, score in zip(beam_indices, token_indices, top_scores):
                    # Check if sequence is finished
                    if token_idx == end_symbol_idx:
                        # Apply length penalty and add to finished sequences
                        normalized_score = score / (curr_length ** length_penalty)
                        finished_sequences.append(curr_tokens[beam_idx].clone())
                        finished_scores.append(normalized_score)
                        continue
                    
                    # Add token to sequence
                    new_seq = torch.cat([curr_tokens[beam_idx], token_idx.unsqueeze(0)], dim=0).unsqueeze(0)
                    new_tokens.append(new_seq)
                    new_scores.append(score)
                
                # Check if all beams are finished or we've reached max length
                if len(new_tokens) == 0 or step == max_length - 1:
                    # Add any remaining beams to finished sequences
                    for beam_idx, score in enumerate(beam_scores):
                        if beam_idx < len(curr_tokens):
                            normalized_score = score / (curr_length ** length_penalty)
                            finished_sequences.append(curr_tokens[beam_idx].clone())
                            finished_scores.append(normalized_score)
                    break
                
                # Update current tokens and scores for next step
                curr_tokens = torch.cat(new_tokens, dim=0)
                beam_scores = torch.tensor(new_scores).to(device)
            
            # Select the sequence with the highest score
            if finished_sequences:
                best_idx = torch.tensor(finished_scores).argmax()
                best_seq = finished_sequences[best_idx]
            else:
                # If no sequence finished, take the first beam
                best_seq = curr_tokens[0]
            
            generated_sequences.append(best_seq)
        
        return generated_sequences


class TransformerForMT(Transformer):
    """
    Transformer model specifically configured for machine translation tasks.
    Includes label smoothing and specialized loss function.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int = 0,
        label_smoothing: float = 0.1,
        **kwargs
    ):
        """
        Initialize the machine translation Transformer.
        
        Args:
            src_vocab_size: Size of source language vocabulary
            tgt_vocab_size: Size of target language vocabulary
            pad_idx: Index of the padding token
            label_smoothing: Label smoothing factor
            **kwargs: Additional arguments for the base Transformer
        """
        super().__init__(src_vocab_size, tgt_vocab_size, **kwargs)
        
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothedCrossEntropy(label_smoothing, tgt_vocab_size, pad_idx)
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with loss calculation.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_length]
            tgt: Target sequence tensor [batch_size, tgt_seq_length]
            src_mask: Optional mask for padding in source
            tgt_mask: Optional mask for padding in target
            
        Returns:
            Tuple of (output logits, loss)
        """
        # Split target into input and output
        # For training, we feed all but the last token and predict all but the first
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create masks
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        
        tgt_input_mask = self.create_padding_mask(tgt_input)
        causal_mask = self.create_causal_mask(tgt_input.size(1)).to(tgt.device)
        
        # Combine masks using multiplication instead of bitwise AND
        tgt_input_mask = tgt_input_mask.float() * causal_mask.float()
        
        # Get model output
        logits = super().forward(src, tgt_input, src_mask, tgt_input_mask)
        
        # Calculate loss
        loss = self.criterion(logits.contiguous().view(-1, self.tgt_vocab_size), tgt_output.contiguous().view(-1))
        
        return logits, loss
    
    def translate(
        self, 
        src: torch.Tensor, 
        start_symbol_idx: int, 
        end_symbol_idx: int, 
        max_length: int = 100,
        beam_size: int = 5
    ) -> List[torch.Tensor]:
        """
        Translate source sequences to target language.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_length]
            start_symbol_idx: Index of the start symbol in target vocabulary
            end_symbol_idx: Index of the end symbol in target vocabulary
            max_length: Maximum length of the generated sequence
            beam_size: Beam size for search (1 for greedy decoding)
            
        Returns:
            List of translated sequences
        """
        if beam_size <= 1:
            return self.greedy_decode(src, start_symbol_idx, end_symbol_idx, max_length)
        else:
            return self.beam_search_decode(src, start_symbol_idx, end_symbol_idx, beam_size, max_length)


class LabelSmoothedCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    This helps prevent the model from being too confident in its predictions.
    """
    
    def __init__(self, smoothing: float, vocab_size: int, pad_idx: int = 0):
        """
        Initialize the label smoothed cross entropy loss.
        
        Args:
            smoothing: Label smoothing factor
            vocab_size: Size of the vocabulary
            pad_idx: Index of the padding token (to ignore)
        """
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the label smoothed cross entropy loss.
        
        Args:
            pred: Prediction logits [batch_size * seq_length, vocab_size]
            target: Target indices [batch_size * seq_length]
            
        Returns:
            Loss value
        """
        # Create a mask to ignore padding tokens
        mask = (target != self.pad_idx).float()
        
        # Count valid (non-padding) tokens
        n_valid = mask.sum()
        
        # Create smoothed targets
        target_dist = torch.zeros_like(pred)
        target_dist.fill_(self.smoothing / (self.vocab_size - 1))  # Distribute smoothing
        target_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)  # Set main target
        
        # Calculate log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Calculate loss and apply mask
        loss = -torch.sum(target_dist * log_probs, dim=-1) * mask
        
        # Return mean loss over valid tokens
        return loss.sum() / n_valid
