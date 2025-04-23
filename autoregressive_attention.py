import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from attention import MultiHeadAttention, PositionwiseFeedForward

class AutoregressiveDecoderBlock(nn.Module):
    """
    Modified Decoder Block for autoregressive models (like GPT).
    
    Unlike the standard Transformer decoder block, this version:
    1. Can optionally disable cross-attention (for decoder-only models)
    2. Is designed for causal/autoregressive generation
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_cross_attention: bool = False):
        """
        Initialize the Autoregressive Decoder Block.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout probability
            use_cross_attention: Whether to use cross-attention (False for decoder-only models like GPT)
        """
        super().__init__()
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Multi-Head Cross-Attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: Optional[torch.Tensor] = None, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Autoregressive Decoder Block.
        
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            encoder_output: Optional output from the encoder [batch_size, src_seq_length, d_model]
            src_mask: Optional mask for the encoder output [batch_size, 1, 1, src_seq_length]
            tgt_mask: Optional mask for the decoder self-attention [batch_size, 1, seq_length, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Masked Multi-Head Self-Attention with residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Multi-Head Cross-Attention with residual connection and layer normalization (if enabled)
        if self.use_cross_attention and encoder_output is not None:
            cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
            x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Position-wise Feed-Forward Network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
