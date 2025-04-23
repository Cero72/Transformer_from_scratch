import os
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
from autoregressive_model import AutoregressiveLM

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class WikiTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=128, max_size=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Load and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_size:
                text = f.read(max_size)  # Read only a portion for faster processing
            else:
                text = f.read()
        
        print(f"Loaded {len(text)/1024:.2f} KB of text from {os.path.basename(file_path)}")
        
        # Tokenize the text
        print("Tokenizing text...")
        self.tokens = tokenizer.encode(text)
        print(f"Text tokenized into {len(self.tokens)} tokens")
        
        # Create chunks of seq_length+1 (input + target)
        print("Creating training chunks...")
        self.chunks = []
        for i in range(0, len(self.tokens) - seq_length):
            self.chunks.append(self.tokens[i:i + seq_length + 1])
        
        print(f"Created {len(self.chunks)} training chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx])

def train(quick_test=True):
    # Data paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')
    
    # Check if data exists
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        print("Please download the WikiText-2 dataset and place it in the correct directory.")
        return
    
    # Model parameters
    vocab_size = 30000
    d_model = 300  # Changed from 256 to 300 to match standard GloVe dimensions
    num_heads = 6  # Changed from 8 to 6 to make it divide evenly into 300
    num_layers = 4
    dropout = 0.1
    
    # Training parameters - adjust based on quick_test flag
    if quick_test:
        print("\n*** QUICK TEST MODE: Using reduced dataset and parameters ***\n")
        batch_size = 8
        num_epochs = 2
        max_train_size = 1024 * 1024  # ~1MB of text for quick testing
        max_valid_size = 100 * 1024   # ~100KB for validation
        vocab_size = 10000
    else:
        batch_size = 16 if torch.cuda.is_available() else 8
        num_epochs = 10
        max_train_size = None  # Use full dataset
        max_valid_size = None
    
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_steps = 1000
    max_steps = 20000  # Reduced from 100000 to 20000 for faster training
    save_every = 1000  # Reduced from 5000 to 1000 to get more checkpoints
    eval_every = 500   # Reduced from 1000 to 500 for more frequent evaluation
    
    # Initialize tokenizer and build vocabulary
    print("Initializing tokenizer and building vocabulary...")
    tokenizer = Tokenizer(vocab_size=vocab_size)
    
    # For quick testing, we'll read a smaller portion of the file
    if quick_test:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read(max_train_size)
        tokenizer.build_vocab([train_text])
    else:
        tokenizer.build_vocab_from_file(train_path)
    
    print(f"Vocabulary built with {len(tokenizer)} tokens")
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = WikiTextDataset(train_path, tokenizer, max_size=max_train_size)
    valid_dataset = WikiTextDataset(valid_path, tokenizer, max_size=max_valid_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Path to pretrained embeddings
    pretrained_embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_embeddings', 'glove.6B.100d.txt')
    
    # Initialize model
    print("Initializing model...")
    model = AutoregressiveLM(
        vocab_size=len(tokenizer),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        embedding_type=pretrained_embeddings_path,  # Use the full path to the embeddings file
        freeze_embeddings=False,  # Allow fine-tuning of embeddings
        pad_idx=tokenizer.PAD_IDX
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/warmup_steps, 1.0))
    
    # Training loop
    global_step = 0
    best_valid_loss = float('inf')
    
    # Create output directory for model checkpoints
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    
    print(f"Starting training on {device}")
    
    # Enable mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    _, loss = model(batch)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            epoch_loss += loss.item()
            
            global_step += 1
            
            # Evaluation
            if global_step % eval_every == 0:
                valid_loss = evaluate(model, valid_loader, device)
                print(f"Step {global_step} | Valid Loss: {valid_loss:.4f}")
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_loss': valid_loss,
                    }, os.path.join(output_dir, 'best_model.pt'))
            
            # Save checkpoint
            if global_step % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': global_step,
                }, os.path.join(output_dir, f'checkpoint_{global_step}.pt'))
            
            # Stop if we've reached max steps
            if global_step >= max_steps:
                break
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        
        if global_step >= max_steps:
            break
    
    print("Training complete!")
    
    # Generate some text examples
    generate_samples(model, tokenizer, device)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, loss = model(batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_samples(model, tokenizer, device, num_samples=2):
    model.eval()
    
    # Sample prompts
    prompts = [
        "The history of artificial intelligence",
        "In recent years, scientists have discovered"
    ]
    
    print("\nGenerated Text Samples:")
    for i, prompt in enumerate(prompts[:num_samples]):
        # Tokenize prompt
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens]).to(device)
        
        # Generate text
        output_ids = model.generate(
            prompt=input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(f"\nSample {i+1}:\nPrompt: {prompt}\nGenerated: {generated_text}\n")

if __name__ == "__main__":
    # Set to True for quick testing, False for full training
    train(quick_test=False)
