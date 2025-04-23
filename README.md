# Building a Transformer from Scratch

This repository contains my implementation of a Transformer-based language model built from scratch using PyTorch. The goal was to understand the inner workings of Transformer models by implementing every component myself rather than using existing libraries.

## What I Learned

- **Attention Mechanisms**: Implemented scaled dot-product attention and multi-head attention, gaining a deep understanding of how self-attention works and why it's so effective for capturing long-range dependencies.

- **Positional Encodings**: Built sinusoidal position embeddings to give the model information about token positions, since attention has no inherent sense of order.

- **Training Dynamics**: Discovered the importance of learning rate scheduling, gradient clipping, and proper initialization for stable training of deep Transformer networks.

- **Embedding Techniques**: Experimented with different embedding strategies including random initialization and pretrained GloVe vectors, learning how to integrate external knowledge into neural networks.

- **Text Generation**: Implemented various decoding strategies (greedy, temperature sampling, top-k, nucleus sampling) and observed their effects on text quality and diversity.

## Model Architecture

I experimented with different configurations and settled on:
- 300-dimensional embeddings (GloVe pretrained)
- 6 attention heads
- 4 transformer layers
- ~27M parameters

## Results

Training on the WikiText-2 dataset (~86M tokens) on an A100 GPU:
- Reached validation loss of 0.384 after ~8,500 steps
- Training took approximately 1-2 hours
- Generated text shows coherent structure but still exhibits some repetition patterns

## Usage

### Training
```bash
python train.py --train_file data/train.txt --valid_file data/valid.txt
```

### Generation
```bash
python generate.py --model_path checkpoints/best_model.pt --prompt "The history of"
```

## Key Takeaways

Building this model helped me understand that:
1. The "magic" of Transformers comes from their ability to process all tokens in parallel while still modeling relationships between any pair of tokens
2. Attention is computationally expensive but incredibly powerful for modeling sequences
3. Even a relatively small Transformer can learn meaningful patterns with the right training approach
4. The biggest challenges were in the implementation details - proper masking, correct dimensionality, and stable training
