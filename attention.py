import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in 'Attention Is All You Need'.
    
    This implementation supports:
    - Regular self-attention
    - Masked self-attention (for decoder)
    - Cross-attention (between encoder and decoder)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the Multi-Head Attention module.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head
        
        # Linear projections for Q, K, V, and output
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # For attention visualization and analysis
        self.attention_weights = None
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        
        Args:
            x: Tensor with shape [batch_size, seq_length, d_model]
            
        Returns:
            Tensor with shape [batch_size, num_heads, seq_length, d_k]
        """
        batch_size, seq_length, _ = x.size()
        
        # Reshape to [batch_size, seq_length, num_heads, d_k]
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        
        # Transpose to [batch_size, num_heads, seq_length, d_k]
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the multi-head attention results.
        
        Args:
            x: Tensor with shape [batch_size, num_heads, seq_length, d_k]
            
        Returns:
            Tensor with shape [batch_size, seq_length, d_model]
        """
        batch_size, _, seq_length, _ = x.size()
        
        # Transpose back to [batch_size, seq_length, num_heads, d_k]
        x = x.transpose(1, 2)
        
        # Combine the heads
        return x.contiguous().view(batch_size, seq_length, self.d_model)
    
    def scaled_dot_product_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_length_q, d_k]
            k: Key tensor [batch_size, num_heads, seq_length_k, d_k]
            v: Value tensor [batch_size, num_heads, seq_length_v, d_k]
            mask: Optional mask tensor [batch_size, 1, 1, seq_length_k] or [batch_size, 1, seq_length_q, seq_length_k]
            
        Returns:
            Tuple of:
                - Output tensor [batch_size, num_heads, seq_length_q, d_k]
                - Attention weights [batch_size, num_heads, seq_length_q, seq_length_k]
        """
        # Calculate attention scores
        # (batch_size, num_heads, seq_length_q, d_k) @ (batch_size, num_heads, d_k, seq_length_k)
        # -> (batch_size, num_heads, seq_length_q, seq_length_k)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale the scores
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Fill masked positions with a very small value
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Calculate the output
        # (batch_size, num_heads, seq_length_q, seq_length_k) @ (batch_size, num_heads, seq_length_v, d_k)
        # -> (batch_size, num_heads, seq_length_q, d_k)
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Attention module.
        
        Args:
            query: Query tensor [batch_size, seq_length_q, d_model]
            key: Key tensor [batch_size, seq_length_k, d_model]
            value: Value tensor [batch_size, seq_length_v, d_model]
            mask: Optional mask tensor [batch_size, 1, seq_length_q, seq_length_k]
            
        Returns:
            Output tensor [batch_size, seq_length_q, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections and split heads
        q = self.split_heads(self.q_projection(query))  # [batch_size, num_heads, seq_length_q, d_k]
        k = self.split_heads(self.k_projection(key))    # [batch_size, num_heads, seq_length_k, d_k]
        v = self.split_heads(self.v_projection(value))  # [batch_size, num_heads, seq_length_v, d_k]
        
        # Scaled dot-product attention
        attn_output, self.attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Combine heads and apply final linear projection
        output = self.combine_heads(attn_output)  # [batch_size, seq_length_q, d_model]
        output = self.output_projection(output)   # [batch_size, seq_length_q, d_model]
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in 'Attention Is All You Need'.
    
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the Position-wise Feed-Forward Network.
        
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Position-wise Feed-Forward Network.
        
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            
        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Apply first linear layer and ReLU activation
        x = F.relu(self.linear1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply second linear layer
        x = self.linear2(x)
        
        return x


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block as described in 'Attention Is All You Need'.
    
    Each block consists of:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    With residual connections and layer normalization after each sub-layer.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the Encoder Block.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Encoder Block.
        
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional mask tensor [batch_size, 1, 1, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Multi-Head Self-Attention with residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Position-wise Feed-Forward Network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block as described in 'Attention Is All You Need'.
    
    Each block consists of:
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention over encoder output
    3. Position-wise Feed-Forward Network
    With residual connections and layer normalization after each sub-layer.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the Decoder Block.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Multi-Head Cross-Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Decoder Block.
        
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            encoder_output: Output from the encoder [batch_size, src_seq_length, d_model]
            src_mask: Optional mask for the encoder output [batch_size, 1, 1, src_seq_length]
            tgt_mask: Optional mask for the decoder self-attention [batch_size, 1, tgt_seq_length, tgt_seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        # Masked Multi-Head Self-Attention with residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Multi-Head Cross-Attention with residual connection and layer normalization
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Position-wise Feed-Forward Network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
