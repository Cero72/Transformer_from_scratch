import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from embedding_layer import PretrainedTransformerEmbedding
from autoregressive_attention import AutoregressiveDecoderBlock

class AutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
        embedding_type: str = 'random',
        freeze_embeddings: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = PretrainedTransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_length=max_seq_length,
            embedding_type=embedding_type,
            dropout=dropout,
            freeze_embeddings=freeze_embeddings
        )
        
        self.decoder_layers = nn.ModuleList([
            AutoregressiveDecoderBlock(d_model, num_heads, d_ff, dropout, use_cross_attention=False)
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  
        return mask
    
    def create_causal_mask(self, seq_length: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  
        return ~mask  
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = self.create_padding_mask(x)
        
        seq_length = x.size(1)
        causal_mask = self.create_causal_mask(seq_length).to(x.device)
        
        combined_mask = padding_mask.float() * causal_mask.float()
        
        x = self.embedding(x)  
        
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, None, combined_mask, None)
        
        logits = self.lm_head(x)  
        
        return logits
    
    def generate(self, 
                prompt: torch.Tensor, 
                max_new_tokens: int = 50, 
                temperature: float = 1.0,
                top_k: int = 0,
                top_p: float = 0.0,
                do_sample: bool = True,
                pad_idx: int = 0,
                eos_idx: Optional[int] = None) -> torch.Tensor:
        self.eval()  
        batch_size = prompt.size(0)
        generated = prompt.clone()
        
        for _ in range(max_new_tokens):
            if generated.size(1) > 1024:  
                input_ids = generated[:, -1024:]
            else:
                input_ids = generated
            
            with torch.no_grad():
                logits = self.forward(input_ids)  
                next_token_logits = logits[:, -1, :]  
            
            next_token_logits = next_token_logits / max(temperature, 1e-8)
            
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)), dim=-1)
                next_token_logits_mask = torch.zeros_like(next_token_logits).scatter_(
                    dim=-1, index=top_k_indices, src=torch.ones_like(top_k_values)
                )
                next_token_logits = torch.where(
                    next_token_logits_mask.bool(),
                    next_token_logits,
                    torch.tensor(float('-inf'), device=next_token_logits.device)
                )
            
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                for batch_idx in range(next_token_logits.size(0)):
                    indices_to_remove[batch_idx] = sorted_indices_to_remove[batch_idx].scatter(
                        dim=0, 
                        index=sorted_indices[batch_idx], 
                        src=sorted_indices_to_remove[batch_idx]
                    )
                
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_idx is not None and (next_token == eos_idx).all():
                break
        
        return generated


class AutoregressiveLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.model = AutoregressiveTransformer(vocab_size=vocab_size, **kwargs)
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    def forward(self, x: torch.Tensor):
        input_ids = x[:, :-1]  
        target_ids = x[:, 1:]  
        
        logits = self.model(input_ids)  
        
        loss = self.criterion(logits.contiguous().view(-1, self.vocab_size), 
                              target_ids.contiguous().view(-1))
        
        return logits, loss
    
    def generate(self, prompt: torch.Tensor, **kwargs):
        return self.model.generate(prompt, pad_idx=self.pad_idx, **kwargs)
