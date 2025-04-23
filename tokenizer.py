import os
import re
import json
import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2, 
                 special_tokens: Optional[Dict[str, str]] = None):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        # Default special tokens
        self.special_tokens = {
            'PAD': '<pad>',
            'UNK': '<unk>',
            'BOS': '<bos>',
            'EOS': '<eos>',
            'MASK': '<mask>'
        }
        
        # Update with user-provided special tokens
        if special_tokens:
            self.special_tokens.update(special_tokens)
        
        # Initialize vocabulary
        self.token2idx = {}
        self.idx2token = {}
        self.vocab_initialized = False
        
        # Add special tokens to vocabulary
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        # Add special tokens to vocabulary with reserved indices
        idx = 0
        for token_type, token in self.special_tokens.items():
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            setattr(self, token_type + '_IDX', idx)  # Set attribute like PAD_IDX, UNK_IDX, etc.
            idx += 1
    
    def build_vocab(self, texts: List[str], char_level: bool = False):
        # Count token frequencies
        counter = Counter()
        
        if char_level:
            # Character-level tokenization
            for text in texts:
                counter.update(text)
        else:
            # Word-level tokenization
            for text in texts:
                tokens = self._tokenize(text)
                counter.update(tokens)
        
        # Sort by frequency and keep most common tokens
        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # Add tokens that meet minimum frequency requirement
        idx = len(self.token2idx)  # Start after special tokens
        for token, count in sorted_tokens:
            if count >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
                
                # Stop if we've reached vocab_size
                if len(self.token2idx) >= self.vocab_size:
                    break
        
        self.vocab_initialized = True
        return self
    
    def build_vocab_from_file(self, file_path: str, char_level: bool = False):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.build_vocab([text], char_level)
    
    def _tokenize(self, text: str) -> List[str]:
        # Simple word-level tokenization
        # This can be extended with more sophisticated tokenization methods
        text = text.lower()  # Lowercase
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        tokens = text.split()
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not self.vocab_initialized:
            raise ValueError("Vocabulary not initialized. Call build_vocab first.")
        
        # Tokenize the text
        if isinstance(text, str):
            if len(text) > 0 and text[0] in self.idx2token.values():  # Check if already tokenized
                tokens = text.split()
            else:
                tokens = self._tokenize(text)
        else:
            tokens = text  # Assume already tokenized
        
        # Add special tokens if requested
        if add_special_tokens:
            tokens = [self.special_tokens['BOS']] + tokens + [self.special_tokens['EOS']]
        
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.token2idx[self.special_tokens['UNK']])
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        if not self.vocab_initialized:
            raise ValueError("Vocabulary not initialized. Call build_vocab first.")
        
        # Convert indices to tokens
        special_token_values = set(self.special_tokens.values())
        tokens = []
        
        for idx in indices:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if skip_special_tokens and token in special_token_values:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.special_tokens['UNK'])
        
        # Join tokens into a string
        if all(len(token) == 1 for token in tokens):  # Character-level
            return ''.join(tokens)
        else:  # Word-level
            return ' '.join(tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = False, 
                     padding: bool = True, truncation: bool = False, 
                     max_length: Optional[int] = None) -> torch.Tensor:
        # Encode each text
        batch_indices = [self.encode(text, add_special_tokens) for text in texts]
        
        # Truncate if needed
        if truncation and max_length is not None:
            batch_indices = [indices[:max_length] for indices in batch_indices]
        
        # Pad sequences if needed
        if padding:
            max_len = max(len(indices) for indices in batch_indices)
            if max_length is not None:
                max_len = min(max_len, max_length)
            
            # Pad with PAD_IDX
            batch_indices = [
                indices + [self.PAD_IDX] * (max_len - len(indices)) 
                for indices in batch_indices
            ]
        
        # Convert to tensor
        return torch.tensor(batch_indices)
    
    def decode_batch(self, batch_indices: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        # Convert tensor to list if needed
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.tolist()
        
        # Decode each sequence
        return [self.decode(indices, skip_special_tokens) for indices in batch_indices]
    
    def save(self, path: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save tokenizer configuration and vocabulary
        config = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': self.special_tokens,
            'token2idx': self.token2idx,
            'vocab_initialized': self.vocab_initialized
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create tokenizer with saved configuration
        tokenizer = cls(
            vocab_size=config['vocab_size'],
            min_freq=config['min_freq'],
            special_tokens=config['special_tokens']
        )
        
        # Restore vocabulary
        tokenizer.token2idx = {k: int(v) for k, v in config['token2idx'].items()}
        tokenizer.idx2token = {int(k): v for k, v in tokenizer.token2idx.items()}
        tokenizer.vocab_initialized = config['vocab_initialized']
        
        # Restore special token indices
        for token_type, token in tokenizer.special_tokens.items():
            if token in tokenizer.token2idx:
                setattr(tokenizer, token_type + '_IDX', tokenizer.token2idx[token])
        
        return tokenizer
    
    def __len__(self):
        return len(self.token2idx)
