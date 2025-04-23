# Transformer Language Model

A PyTorch implementation of a Transformer-based autoregressive language model. This project includes a complete implementation of the Transformer architecture with self-attention mechanisms, positional encodings, and support for pretrained embeddings.

## Features

- **Autoregressive Transformer**: Implementation of a decoder-only Transformer model for language modeling
- **Attention Mechanisms**: Multi-head self-attention with masking for autoregressive generation
- **Pretrained Embeddings**: Support for loading pretrained word embeddings (e.g., GloVe)
- **Positional Encodings**: Sinusoidal position embeddings as described in "Attention Is All You Need"
- **Text Generation**: Sampling strategies including temperature scaling, top-k, and nucleus (top-p) sampling

## Model Architecture

- Embedding dimension: 300
- Attention heads: 6
- Transformer layers: 4
- Parameters: ~27M

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- NumPy

### Installation

```bash
git clone https://github.com/yourusername/transformer-language-model.git
cd transformer-language-model
pip install -r requirements.txt
```

## Usage

### Training

To train the model on your own text dataset:

```bash
python train.py --train_file data/train.txt --valid_file data/valid.txt --output_dir checkpoints
```

Options:

```
--vocab_size      Size of vocabulary (default: 30000)
--d_model         Model dimension (default: 300)
--num_heads       Number of attention heads (default: 6)
--num_layers      Number of transformer layers (default: 4)
--dropout         Dropout rate (default: 0.1)
--batch_size      Batch size (default: 16)
--epochs          Number of epochs (default: 10)
--learning_rate   Learning rate (default: 5e-5)
--embedding_path  Path to pretrained embeddings file (optional)
--quick_test      Run in quick test mode with reduced dataset
```

### Text Generation

To generate text using a trained model:

```bash
python generate.py --model_path checkpoints/best_model.pt --tokenizer_path checkpoints/tokenizer.json --prompt "The history of artificial intelligence"
```

Options:

```
--interactive    Run in interactive mode to input multiple prompts
--temperature    Sampling temperature (default: 0.7)
--top_k          Top-k sampling parameter (default: 40)
--top_p          Top-p sampling parameter (default: 0.9)
--max_length     Maximum length of generated text (default: 100)
```

## Project Structure

```
├── attention.py                 # Attention mechanisms implementation
├── autoregressive_attention.py  # Autoregressive decoder block
├── autoregressive_model.py      # Autoregressive transformer model
├── embedding_layer.py           # Embedding layer with positional encodings
├── tokenizer.py                 # Tokenizer implementation
├── train.py                     # Training script
├── generate.py                  # Text generation script
└── requirements.txt             # Project dependencies
```

## Training Results

The model was trained on the WikiText-2 dataset (~86M tokens) using an A100 GPU. Training achieved a validation loss of 0.384 after approximately 8,500 steps.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) - Pretrained word embeddings
