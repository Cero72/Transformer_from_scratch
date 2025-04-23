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

def load_model(model_path, tokenizer):
    # Create a new model with the same architecture
    model = AutoregressiveLM(
        vocab_size=len(tokenizer),
        d_model=300,
        num_heads=6,
        num_layers=4,
        dropout=0.1,
        embedding_type='random',  # Start with random embeddings
        pad_idx=tokenizer.PAD_IDX
    )
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load the model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model weights!")
        return model
    else:
        print("Error: 'model_state_dict' not found in checkpoint")
        return None

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=40, top_p=0.9, device='cuda'):
    if model is None:
        print("Cannot generate text: model not loaded")
        return ""
    
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
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
    parser = argparse.ArgumentParser(description='Generate text using a trained transformer model')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='checkpoints/tokenizer.json', help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='The history of artificial intelligence', help='Text prompt')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, tokenizer)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            prompt = input("\nEnter a prompt: ")
            if prompt.lower() == 'exit':
                break
            
            print("\nGenerating...")
            generated_text = generate_text(
                model, tokenizer, prompt, 
                args.max_length, args.temperature, args.top_k, args.top_p, args.device
            )
            
            print(f"\nGenerated text:\n{generated_text}")
    else:
        # Generate text from prompt
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        generated_text = generate_text(
            model, tokenizer, args.prompt, 
            args.max_length, args.temperature, args.top_k, args.top_p, args.device
        )
        
        print(f"\nGenerated text:\n{generated_text}")

if __name__ == '__main__':
    main()
