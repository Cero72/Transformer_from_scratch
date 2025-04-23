import os
import torch
import json
import argparse
from tokenizer import Tokenizer
from autoregressive_model import AutoregressiveLM

def load_tokenizer(tokenizer_path):
    # Load tokenizer configuration from JSON file
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Create tokenizer instance
    tokenizer = Tokenizer(vocab_size=config['vocab_size'])
    
    # Restore tokenizer state
    tokenizer.token2idx = config['token2idx']
    tokenizer.idx2token = {idx: token for token, idx in config['token2idx'].items()}
    tokenizer.special_tokens = config['special_tokens']
    tokenizer.vocab_initialized = config['vocab_initialized']
    
    # Set special token indices
    for token_name, token in tokenizer.special_tokens.items():
        setattr(tokenizer, f"{token_name}_IDX", tokenizer.token2idx.get(token, 0))
    
    return tokenizer

def load_model(model_path, tokenizer_path):
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Get model parameters from checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract model configuration from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Determine model parameters from state dict
    vocab_size = len(tokenizer)
    d_model = state_dict['model.embedding.token_embedding.weight'].size(1)  # 300
    
    # Create model with same architecture
    model = AutoregressiveLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=6,  # Same as training
        num_layers=4,  # Same as training
        dropout=0.1,   # Can be lower for inference
        pad_idx=tokenizer.PAD_IDX
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    
    return model, tokenizer

def generate(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, device='cuda'):
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens]).to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            prompt=input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained Transformer model')
    parser.add_argument('--model_path', type=str, default='model_checkpoints/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='model_checkpoints/tokenizer.json', help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='The history of artificial intelligence', help='Text prompt')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, args.tokenizer_path)
    print(f"Model loaded. Vocabulary size: {len(tokenizer)}")
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            prompt = input("\nEnter a prompt: ")
            if prompt.lower() == 'exit':
                break
            
            print("\nGenerating...")
            generated_text = generate(
                model, tokenizer, prompt, 
                args.max_length, args.temperature, args.top_k, args.top_p, args.device
            )
            
            print(f"\nGenerated text:\n{generated_text}")
    else:
        # Generate text from prompt
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        generated_text = generate(
            model, tokenizer, args.prompt, 
            args.max_length, args.temperature, args.top_k, args.top_p, args.device
        )
        
        print(f"\nGenerated text:\n{generated_text}")

if __name__ == '__main__':
    main()
