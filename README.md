# GPT-2 From Scratch

This is a small learning project for building and training a GPT-2 style language model from scratch in PyTorch. It implements the main pieces of a decoder-only Transformer: token and positional embeddings, causal self-attention, Transformer blocks, layer normalization, feed-forward layers, next-token training, and simple greedy text generation.


## Project Structure

| File | Purpose |
| --- | --- |
| `Attention_Module.py` | Implements causal self-attention and multi-head attention modules. |
| `model.py` | Defines the GPT configuration, Transformer block, GPT model, GELU, LayerNorm, feed-forward network, and simple text generation. |
| `gpt2_dataloader.py` | Builds tokenized input/target pairs for next-token prediction using the GPT-2 tokenizer. |
| `train.py` | Trains the model, evaluates train/validation loss, prints generated samples, and saves a loss plot. |
| `data_extraction.py` | Downloads plain-text training data from the Wikipedia API into `train_data.txt` when needed. |
| `train_data.txt` | Local text corpus used for training. |
| `requirements.txt` | Python dependencies. |

## Model Overview

The model in `model.py` uses the following default configuration:

```python
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

The architecture is GPT-like:

- GPT-2 tokenizer vocabulary size: `50257`
- learned token embeddings
- learned positional embeddings
- stacked decoder-only Transformer blocks
- causal multi-head self-attention
- feed-forward network with GELU activation
- residual connections and layer normalization
- final linear output head for next-token logits

## Training Setup

Training uses next-token prediction with teacher forcing.

In `gpt2_dataloader.py`, each training sample is created as:

```python
input_chunk = token_ids[i:i+max_length]
target_chunk = token_ids[i+1: i+max_length+1]
```

So the model receives the real previous tokens and learns to predict the next token at every position.

In `train.py`, the model output is compared against the shifted target sequence:

```python
logits = model(input_batch)
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),
    target_batch.flatten()
)
```

The script splits `train_data.txt` into:

- 90% training text
- 10% validation text

It trains with AdamW, prints train/validation loss every few steps, generates a sample after each epoch, and saves a loss plot to `loss-plot.pdf`.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is required by the code but is not currently listed in `requirements.txt`. Install the version that matches your system from the official PyTorch instructions:

```bash
pip install torch
```

`data_extraction.py` also uses `requests`, which is not currently listed in `requirements.txt`:

```bash
pip install requests
```

## Prepare Data

The repository includes `train_data.txt`. To regenerate it from the Wikipedia API, run:

```bash
python data_extraction.py
```

The current extraction script downloads the plain-text Wikipedia article for "Adolf Hitler" and writes it to `train_data.txt`.

## Run a Model Smoke Test

To instantiate the model and run simple greedy generation from an untrained model:

```bash
python model.py
```

This does not train the model. It only checks that the model can run a forward generation loop.

## Train the Model

Run:

```bash
python train.py
```

The training script will:

- load `train_data.txt`
- create training and validation dataloaders
- train the GPT model for 10 epochs
- evaluate loss periodically
- print generated text samples
- save a loss plot as `loss-plot.pdf`

## Current Notes and Limitations

- `gpt2_dataloader.py` contains example/demo code at module level, so importing `dataloader_v1` also runs that demo and prints an embedding shape. Moving the demo under `if __name__ == "__main__":` would make imports cleaner.
- `train.py` currently trains a full 12-layer, 768-dimensional GPT-style model, which can be slow or memory-heavy on CPU.
- The training corpus is very small for this model size, so generated text quality will be limited.
- Generation uses greedy decoding with `argmax`; there is no temperature, top-k, or nucleus sampling yet.
- `train.py` saves a loss plot, but model checkpoint saving should be reviewed before relying on the output file.

## Reference

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
