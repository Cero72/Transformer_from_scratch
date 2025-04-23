import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datasets import load_dataset
from tokenizer import Tokenizer
from autoregressive_model import AutoregressiveLM

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class WikiTextDataset(Dataset):
    """
    Dataset for word-level language modeling on the WikiText-2 dataset.
    """
    
    def __init__(self, texts, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Concatenate all texts and tokenize
        full_text = ' '.join(texts)
        self.tokens = tokenizer.encode(full_text)
        
        # Create chunks of seq_length+1 (input + target)
        self.chunks = []
        for i in range(0, len(self.tokens) - seq_length):
            chunk = self.tokens[i:i + seq_length + 1]
            self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx])

def main():
    # Load WikiText-2 dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    # Create a tokenizer
    tokenizer = Tokenizer()
    
    # Build vocabulary from training data
    train_texts = dataset["train"]["text"]
    tokenizer.build_vocab_from_texts(train_texts)
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens")
    
    # Add special tokens if not already present
    special_tokens = ['<pad>', '<bos>', '<eos>']
    for token in special_tokens:
        if token not in tokenizer.vocab:
            tokenizer.vocab[token] = len(tokenizer.vocab)
            tokenizer.id_to_token[len(tokenizer.id_to_token)] = token
    
    # Create datasets
    train_dataset = WikiTextDataset(dataset["train"]["text"], tokenizer, seq_length=128)
    val_dataset = WikiTextDataset(dataset["validation"]["text"], tokenizer, seq_length=128)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    vocab_size = len(tokenizer.vocab)
    model = AutoregressiveLM(
        vocab_size=vocab_size,
        d_model=384,      # Model dimension
        num_heads=8,      # Number of attention heads
        num_layers=6,     # Number of decoder layers
        d_ff=1536,        # Feed-forward dimension (4x d_model)
        dropout=0.1,      # Dropout rate
        max_seq_length=128,
        pad_idx=tokenizer.vocab.get('<pad>', 0)
    )
    
    # Training parameters
    num_epochs = 10       # Number of training epochs
    learning_rate = 5e-5  # Learning rate
    weight_decay = 0.01   # Weight decay for regularization
    warmup_steps = 1000   # Learning rate warmup steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"Training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    best_val_loss = float('inf')
    total_steps = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            _, loss = model(batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            total_steps += 1
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, loss = model(batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'wikitext_model.pt')
            print("Model saved!")
    
    # Generate text samples
    model.eval()
    
    # Define prompts
    prompts = [
        "The history of artificial intelligence",
        "In recent years, scientists have discovered",
        "The main challenge in developing",
        "According to the latest research"
    ]
    
    print("\nGenerating text samples:")
    for prompt in prompts:
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens]).to(device)
        
        # Generate continuation
        output_ids = model.generate(
            prompt=input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        
        # Decode and print
        output_text = tokenizer.decode(output_ids[0].tolist())
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{output_text}'\n")

if __name__ == "__main__":
    main()
