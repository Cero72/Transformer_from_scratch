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

def create_model(tokenizer):
    # Create a new model with the same architecture as the trained model
    model = AutoregressiveLM(
        vocab_size=len(tokenizer),
        d_model=300,         # Same as training
        num_heads=6,         # Same as training
        num_layers=4,        # Same as training
        dropout=0.1,         # Same as training
        pad_idx=tokenizer.PAD_IDX
    )
    
    return model

def generate(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, device='cpu'):
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
    parser = argparse.ArgumentParser(description='Generate text using a new Transformer model')
    parser.add_argument('--tokenizer_path', type=str, default='model_checkpoints/tokenizer.json', help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='The history of artificial intelligence', help='Text prompt')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}")
    
    # Create model
    print("Creating new model...")
    model = create_model(tokenizer)
    print(f"Model created. Size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerating...")
    generated_text = generate(
        model, tokenizer, args.prompt, 
        args.max_length, args.temperature, args.top_k, args.top_p, args.device
    )
    
    print(f"\nGenerated text:\n{generated_text}")

if __name__ == '__main__':
    main()
