import os
import torch

# Path to the model file
model_path = 'model_checkpoints/best_model.pt'

# Check if the file exists
if os.path.exists(model_path):
    print(f"Model file exists at {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    
    # Try to read the first few bytes to see if it's a valid file
    with open(model_path, 'rb') as f:
        header = f.read(10)
        print(f"First 10 bytes: {header}")
    
    try:
        # Try to load the model using torch.load
        print("Attempting to load the model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # If successful, print the keys in the checkpoint
        print("\nSuccessfully loaded the model!")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # If 'model_state_dict' is in the keys, print the keys in the state dict
        if 'model_state_dict' in checkpoint:
            print(f"\nModel state dict keys: {list(checkpoint['model_state_dict'].keys())[:5]}... (truncated)")
            
    except Exception as e:
        print(f"\nError loading model: {e}")
        
        # If there's an error, try to determine if it's a text file
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print(f"\nThis appears to be a text file. First line: {first_line[:100]}...")
        except UnicodeDecodeError:
            print("\nThis is not a text file (binary data)")
else:
    print(f"Model file does not exist at {model_path}")
