#!/usr/bin/env python3
"""
GPT-style transformer training script for small Tang poem dataset.
Quick training for testing and inference.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import time
import math

# Configuration for small dataset
CONFIG = {
    'dataset': 'tang_small',
    'batch_size': 16,  # Small batch size for quick training
    'block_size': 128,  # Smaller context window
    'max_iters': 1000,  # Fewer iterations for quick test
    'eval_interval': 50,
    'learning_rate': 1e-3,
    'eval_iters': 20,
    'n_embd': 256,  # Smaller model
    'n_head': 4,
    'n_layer': 4,
    'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")
print(f"Configuration: {CONFIG}")

# Load the dataset
def load_data():
    data = torch.load(f"{CONFIG['dataset']}_train.bin", map_location=CONFIG['device'])
    with open(f"{CONFIG['dataset']}_meta.pkl", 'rb') as f:
        meta = pickle.load(f)
    return data, meta

data, meta = load_data()
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']
print(f"Dataset size: {len(data):,} tokens")
print(f"Vocabulary size: {vocab_size}")

# Import the model classes from the separate module
from gpt_model import GPTModel

# Create model
model = GPTModel(
    vocab_size=vocab_size,
    n_embd=CONFIG['n_embd'],
    n_head=CONFIG['n_head'],
    n_layer=CONFIG['n_layer'],
    block_size=CONFIG['block_size'],
    dropout=CONFIG['dropout']
).to(CONFIG['device'])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - CONFIG['block_size'], (CONFIG['batch_size'],))
    x = torch.stack([data[i:i+CONFIG['block_size']] for i in ix])
    y = torch.stack([data[i+1:i+CONFIG['block_size']+1] for i in ix])
    x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
    return x, y

# Loss estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(CONFIG['eval_iters'])
        for k in range(CONFIG['eval_iters']):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print("Starting training...")
for iter in range(CONFIG['max_iters']):
    
    # Every once in a while we estimate the loss on train and val sets
    if iter % CONFIG['eval_interval'] == 0 or iter == CONFIG['max_iters'] - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), 'gpt_tang_small_model.pt')
print("Training completed! Model saved as 'gpt_tang_small_model.pt'")

# Generate a sample poem
print("\nGenerating a sample poem...")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=CONFIG['device'])
context[0, 0] = stoi['春']  # Start with '春' (spring)
generated = model.generate(context, max_new_tokens=50, temperature=0.8, top_k=200)
generated_text = ''.join([itos[i.item()] for i in generated[0]])
print(f"Generated poem:\n{generated_text}") 