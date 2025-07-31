#!/usr/bin/env python3
"""
Training script for medium Tang poem dataset with improved GPT model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import math
import time
import argparse

# Import the improved model classes
from gpt_model_improved import GPTModel

# Configuration for medium dataset
CONFIG = {
    'dataset': 'tang_medium',
    'batch_size': 32,  # Increased from 16
    'block_size': 256,  # Increased from 128
    'max_iters': 2000,  # Increased from 1000
    'eval_interval': 100,  # Increased from 50
    'learning_rate': 3e-4,  # Reduced from 1e-3 for better stability
    'eval_iters': 50,  # Increased from 20
    'n_embd': 512,  # Increased from 256
    'n_head': 8,  # Increased from 4
    'n_layer': 6,  # Increased from 4
    'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def get_batch(data, block_size, batch_size, device):
    """Generate a small batch of data of inputs x and targets y."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device):
    """Estimate loss on train and val sets."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def main():
    print(f"Using device: {CONFIG['device']}")
    print(f"Configuration: {CONFIG}")
    
    # Load the dataset
    print(f"Loading {CONFIG['dataset']} dataset...")
    data = torch.load(f'{CONFIG["dataset"]}_train.bin')
    meta = pickle.load(open(f'{CONFIG["dataset"]}_meta.pkl', 'rb'))
    
    vocab_size = meta['vocab_size']
    itos = meta['itos']
    stoi = meta['stoi']
    
    print(f"Dataset size: {len(data):,} tokens")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        n_embd=CONFIG['n_embd'],
        n_head=CONFIG['n_head'],
        n_layer=CONFIG['n_layer'],
        block_size=CONFIG['block_size'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # Print model parameters
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for iter in range(CONFIG['max_iters']):
        # Every once in a while evaluate the loss on train and val sets
        if iter % CONFIG['eval_interval'] == 0 or iter == CONFIG['max_iters'] - 1:
            losses = estimate_loss(model, data, CONFIG['block_size'], CONFIG['batch_size'], 
                                 CONFIG['eval_iters'], CONFIG['device'])
            print(f"Step {iter}: train loss {losses:.4f}")
        
        # Sample a batch of data
        xb, yb = get_batch(data, CONFIG['block_size'], CONFIG['batch_size'], CONFIG['device'])
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds!")
    
    # Save the model
    model_path = f'gpt_tang_medium_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")
    
    # Generate a sample poem
    print("Generating a sample poem...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=CONFIG['device'])
    context[0, 0] = stoi['春']  # Start with '春'
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=50)
    
    generated_text = ''.join([itos[i.item()] for i in generated[0]])
    print("Generated poem:")
    print(generated_text)

if __name__ == "__main__":
    main() 