#!/usr/bin/env python3
"""
Generation script for the small GPT Tang poem model.
Quick testing and inference.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import argparse
import random
import math

# Import the model classes from the separate module
from gpt_model import GPTModel

def load_model(model_path, meta_path):
    """Load a trained GPT model and metadata."""
    # Load metadata
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    vocab_size = meta['vocab_size']
    
    # Model configuration (should match training)
    n_embd = 256
    n_head = 4
    n_layer = 4
    block_size = 128
    dropout = 0.1
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, stoi, itos

def generate_poem(model, stoi, itos, starting_text="", max_new_tokens=100, temperature=0.8, top_k=200):
    """Generate a poem using the GPT model."""
    model.eval()
    
    # Prepare context
    if starting_text:
        # Encode the starting text
        context = torch.tensor([[stoi[c] for c in starting_text]], dtype=torch.long)
    else:
        # Start with a random character from common poem starters
        starters = ['春', '秋', '月', '花', '山', '水', '风', '雨', '夜', '日']
        start_char = random.choice(starters)
        context = torch.tensor([[stoi[start_char]]], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            context, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode
    generated_text = ''.join([itos[i.item()] for i in generated[0]])
    
    return generated_text

def format_poem(text, max_line_length=7):
    """Format the generated text into proper poem lines."""
    lines = []
    current_line = ""
    
    for char in text:
        current_line += char
        if len(current_line) >= max_line_length:
            lines.append(current_line)
            current_line = ""
    
    if current_line:
        lines.append(current_line)
    
    return '\n'.join(lines)

def interactive_generation(model, stoi, itos):
    """Interactive poem generation."""
    print("🎭 Tang Poem AI Generator (Small GPT Model)")
    print("=" * 50)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help")
    print("  'params' - Show current generation parameters")
    print("  Just type a starting text to generate a poem")
    print()
    
    # Default parameters
    params = {
        'temperature': 0.8,
        'top_k': 200,
        'max_tokens': 100
    }
    
    while True:
        try:
            user_input = input("🎨 Enter starting text (or command): ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'help':
                print("Commands: quit/exit, help, params")
                print("Or just type Chinese characters to start a poem!")
                continue
            elif user_input.lower() == 'params':
                print(f"Current parameters: {params}")
                continue
            elif not user_input:
                user_input = ""  # Generate with random starter
            
            print(f"\n🎯 Generating poem starting with: '{user_input if user_input else 'random starter'}'")
            print("⏳ Please wait...")
            
            # Generate poem
            poem = generate_poem(
                model, stoi, itos,
                starting_text=user_input,
                max_new_tokens=params['max_tokens'],
                temperature=params['temperature'],
                top_k=params['top_k']
            )
            
            # Format and display
            formatted_poem = format_poem(poem)
            print(f"\n📜 Generated Poem:")
            print("=" * 30)
            print(formatted_poem)
            print("=" * 30)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Generate Tang poems with small GPT model')
    parser.add_argument('--model', default='gpt_tang_small_model.pt', help='Path to model file')
    parser.add_argument('--meta', default='tang_small_meta.pkl', help='Path to metadata file')
    parser.add_argument('--start', default='', help='Starting text for generation')
    parser.add_argument('--tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print(f"Loading model from {args.model}...")
        model, stoi, itos = load_model(args.model, args.meta)
        print("✅ Model loaded successfully!")
        
        if args.interactive:
            interactive_generation(model, stoi, itos)
        else:
            # Single generation
            print(f"Generating poem starting with: '{args.start if args.start else 'random starter'}'")
            poem = generate_poem(
                model, stoi, itos,
                starting_text=args.start,
                max_new_tokens=args.tokens,
                temperature=args.temp,
                top_k=args.top_k
            )
            
            formatted_poem = format_poem(poem)
            print(f"\n📜 Generated Poem:")
            print("=" * 30)
            print(formatted_poem)
            print("=" * 30)
    
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file {e.filename}")
        print("Make sure you have trained the model first using train_gpt_small.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 