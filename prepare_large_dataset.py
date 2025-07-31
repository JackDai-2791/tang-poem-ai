#!/usr/bin/env python3
"""
Prepare script for the large Tang poem dataset.
"""

import torch
import pickle
import os

def prepare_large_dataset(input_file="tang_poems_large.txt", output_prefix="tang_large"):
    """Prepare the large Tang poem dataset for training."""
    
    print(f"Reading data from {input_file}...")
    
    # Read the data
    with open(input_file, encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text):,} characters")
    
    # Build the vocabulary (unique characters)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Unique characters: {chars}")
    print(f"Vocab size: {vocab_size}")
    
    # Mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the entire text
    print("Encoding dataset...")
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Encoded data length: {len(data):,} tokens")
    
    # Save the metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    print(f"Saving metadata to {output_prefix}_meta.pkl...")
    with open(f'{output_prefix}_meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    # Save the encoded data
    print(f"Saving encoded data to {output_prefix}_train.bin...")
    torch.save(data, f'{output_prefix}_train.bin')
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total characters: {len(text):,}")
    print(f"  Total tokens: {len(data):,}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Average tokens per character: {len(data) / len(text):.2f}")
    
    # Test encoding/decoding
    print("\nSample encoding/decoding test:")
    sample_text = text[:100]
    encoded = [stoi[c] for c in sample_text]
    decoded = ''.join([itos[i] for i in encoded])
    print(f"  Original: {sample_text}")
    print(f"  Encoded: {encoded[:20]}...")
    print(f"  Decoded: {decoded}")
    print(f"  Match: {sample_text == decoded}")
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"Files created:")
    print(f"  - {output_prefix}_meta.pkl (metadata)")
    print(f"  - {output_prefix}_train.bin (encoded data)")

if __name__ == "__main__":
    prepare_large_dataset() 