import torch
import torch.nn as nn
import math
import numpy as np
import os

class PretrainedTransformerEmbedding(nn.Module):
    """
    Transformer embedding layer with optional pretrained word embeddings:
    1. Token embeddings: vectors loaded from file or randomly initialized
    2. Positional encodings: fixed patterns based on position in the sequence
    """
    
    def __init__(self, vocab_size, d_model, embedding_type='random', max_seq_length=5000, dropout=0.1, freeze_embeddings=False, word_to_idx=None):
        """
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the embedding vectors
            embedding_type (str): Type of embeddings ('random' or path to custom embeddings file)
            max_seq_length (int): Maximum sequence length to support
            dropout (float): Dropout rate applied to embeddings
            freeze_embeddings (bool): Whether to freeze the pretrained embeddings during training
            word_to_idx (dict): Mapping from words to indices in your vocabulary
        """
        super(PretrainedTransformerEmbedding, self).__init__()
        
        # Token embedding layer: maps token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Load pretrained embeddings if specified
        if embedding_type != 'random' and os.path.exists(embedding_type):
            self._load_pretrained_embeddings(embedding_type, vocab_size, d_model, freeze_embeddings, word_to_idx)
        else:
            if embedding_type != 'random':
                print(f"Unknown embedding type or file not found: {embedding_type}. Using randomly initialized embeddings.")
        
        # Register positional encoding buffer (not a parameter to be learned)
        self.register_buffer("positional_encoding", self._create_positional_encoding(max_seq_length, d_model))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Save embedding dimension for scaling
        self.d_model = d_model
    
    def _load_pretrained_embeddings(self, pretrained_path, vocab_size, d_model, freeze_embeddings, word_to_idx=None, is_binary=False):
        """
        Loads pretrained word embeddings from file.
        
        Args:
            pretrained_path (str): Path to pretrained embeddings file
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the embedding vectors
            freeze_embeddings (bool): Whether to freeze the embeddings during training
            word_to_idx (dict): Mapping from words to indices in your vocabulary
            is_binary (bool): Whether the embeddings file is in binary format
        """
        try:
            print(f"Loading pretrained embeddings from {pretrained_path}")
            
            # Initialize embeddings with random values
            embeddings = torch.randn(vocab_size, d_model)
            
            # Load pretrained embeddings from file
            if not is_binary:  # Text format (like GloVe)
                word_to_vec = {}
                with open(pretrained_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        values = line.strip().split(' ')
                        word = values[0]
                        vector = torch.tensor([float(val) for val in values[1:]])
                        word_to_vec[word] = vector
                
                # Map to our vocabulary if provided
                if word_to_idx is not None:
                    for word, idx in word_to_idx.items():
                        if word in word_to_vec:
                            vector = word_to_vec[word]
                            
                            # Handle dimension mismatch
                            if vector.size(0) < d_model:
                                # Pad with zeros if pretrained dimension is smaller
                                vector = torch.cat([vector, torch.zeros(d_model - vector.size(0))])
                            elif vector.size(0) > d_model:
                                # Truncate if pretrained dimension is larger
                                vector = vector[:d_model]
                            
                            embeddings[idx] = vector
            
            # Load the pretrained embeddings into our embedding layer
            self.token_embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze_embeddings)
            print(f"Successfully loaded pretrained embeddings. Embeddings {'frozen' if freeze_embeddings else 'will be fine-tuned'}.")
            
        except Exception as e:
            print(f"Error loading pretrained embeddings: {e}")
            print("Using randomly initialized embeddings instead.")
    
    def _create_positional_encoding(self, max_seq_length, d_model):
        """
        Creates positional encodings for the inputs.
        Formula from the Attention is All You Need paper:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            max_seq_length (int): Maximum sequence length to support
            d_model (int): Dimension of the embedding vectors
            
        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, max_seq_length, d_model)
        """
        # Initialize a matrix for positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a vector of positions: [0, 1, 2, ..., max_seq_length-1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate number of even indices in d_model
        num_even_indices = d_model // 2 + d_model % 2
        num_odd_indices = d_model // 2
        
        # Create a vector for division term
        div_term = torch.exp(torch.arange(0, num_even_indices).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term[:d_model//2 + d_model%2])
        
        # Apply cosine to odd indices (if they exist)
        if d_model > 1:  # Only if we have odd indices
            div_term_odd = div_term[:num_odd_indices]
            pe[:, 1::2] = torch.cos(position * div_term_odd)
        
        # Add batch dimension [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        return pe
    
    def forward(self, x):
        """
        Forward pass through the embedding layer.
        
        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Embedded representation with positional encoding
                         of shape (batch_size, seq_length, d_model)
        """
        seq_length = x.size(1)
        
        # Get token embeddings
        token_embeddings = self.token_embedding(x)
        
        # Scale embeddings (as per the paper)
        token_embeddings = token_embeddings * math.sqrt(self.d_model)
        
        # Add positional encodings (using only the needed length)
        embeddings = token_embeddings + self.positional_encoding[:, :seq_length, :]
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings

# Example usage:
if __name__ == "__main__":
    # Define hyperparameters
    vocab_size = 10000  # Size of the vocabulary
    d_model = 300       # Dimension of the embedding (common for GloVe/Word2Vec)
    batch_size = 16     # Batch size
    seq_length = 30     # Sequence length
    
    # Create embedding layer
    embedding_layer = PretrainedTransformerEmbedding(
        vocab_size, 
        d_model, 
        embedding_type='random',
        freeze_embeddings=True  # Set to False to fine-tune embeddings
    )
    
    # Create fake input (batch of token IDs)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Pass through embedding layer
    embedded = embedding_layer(inputs)
    
    # Print shape of output
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {embedded.shape}")