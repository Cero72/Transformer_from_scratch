import os
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
from autoregressive_model import AutoregressiveLM

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TextDataset(Dataset):
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

def train(args):
    # Data paths
    train_path = args.train_file
    valid_path = args.valid_file
    
    # Check if data exists
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        return
    
    # Model parameters
    vocab_size = args.vocab_size
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    dropout = args.dropout
    
    # Training parameters
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    save_every = args.save_every
    eval_every = args.eval_every
    
    # Quick test mode
    if args.quick_test:
        print("\n*** QUICK TEST MODE: Using reduced dataset and parameters ***\n")
        max_train_size = 1024 * 1024  # ~1MB of text for quick testing
        max_valid_size = 100 * 1024   # ~100KB for validation
    else:
        max_train_size = None  # Use full dataset
        max_valid_size = None
    
    # Initialize tokenizer and build vocabulary
    print("Initializing tokenizer and building vocabulary...")
    tokenizer = Tokenizer(vocab_size=vocab_size)
    
    # For quick testing, we'll read a smaller portion of the file
    if args.quick_test:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read(max_train_size)
        tokenizer.build_vocab([train_text])
    else:
        tokenizer.build_vocab_from_file(train_path)
    
    print(f"Vocabulary built with {len(tokenizer)} tokens")
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = TextDataset(train_path, tokenizer, max_size=max_train_size)
    valid_dataset = TextDataset(valid_path, tokenizer, max_size=max_valid_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Path to pretrained embeddings if provided
    embedding_path = args.embedding_path
    
    # Initialize model
    print("Initializing model...")
    model = AutoregressiveLM(
        vocab_size=len(tokenizer),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        embedding_type=embedding_path if embedding_path else 'random',
        freeze_embeddings=args.freeze_embeddings,
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
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    
    print(f"Starting training on {device}")
    
    # Enable mixed precision training if available
    scaler = torch.cuda.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Mixed precision training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
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
            if max_steps and global_step >= max_steps:
                break
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        
        if max_steps and global_step >= max_steps:
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

def main():
    parser = argparse.ArgumentParser(description='Train a transformer language model')
    
    # Data parameters
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--valid_file', type=str, required=True, help='Path to validation data file')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=30000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=300, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Embedding parameters
    parser.add_argument('--embedding_path', type=str, default=None, 
                        help='Path to pretrained embeddings file (optional)')
    parser.add_argument('--freeze_embeddings', action='store_true', 
                        help='Freeze pretrained embeddings during training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for learning rate scheduler')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of training steps')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--eval_every', type=int, default=500, help='Evaluate every N steps')
    
    # Other parameters
    parser.add_argument('--quick_test', action='store_true', help='Run in quick test mode with reduced dataset')
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
