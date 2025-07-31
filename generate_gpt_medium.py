#!/usr/bin/env python3
"""
Generation script for medium Tang poem GPT model.
"""

import torch
import pickle
import argparse
from gpt_model_improved import GPTModel

CONFIG = {
    'dataset': 'tang_medium',
    'batch_size': 32,
    'block_size': 256,
    'n_embd': 512,
    'n_head': 8,
    'n_layer': 6,
    'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_model(model_path, meta_path):
    """Load the trained model and metadata."""
    print(f"Loading model from {model_path}...")
    
    # Load metadata
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    vocab_size = meta['vocab_size']
    itos = meta['itos']
    stoi = meta['stoi']
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        n_embd=CONFIG['n_embd'],
        n_head=CONFIG['n_head'],
        n_layer=CONFIG['n_layer'],
        block_size=CONFIG['block_size'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, itos, stoi

def generate_poem(model, itos, stoi, start_text="春", max_tokens=100, temperature=0.8, top_k=50):
    """Generate a poem starting with the given text."""
    print(f"Generating poem starting with: '{start_text}'")
    
    # Encode the starting text
    context = torch.tensor([[stoi[c] for c in start_text]], dtype=torch.long, device=CONFIG['device'])
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            context, 
            max_new_tokens=max_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode
    generated_text = ''.join([itos[i.item()] for i in generated[0]])
    
    return generated_text

def interactive_mode(model, itos, stoi):
    """Interactive poem generation mode."""
    print("🎭 Tang Poem AI Generator (Medium GPT Model)")
    print("=" * 50)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help")
    print("  'params' - Show current generation parameters")
    print("  Just type a starting text to generate a poem")
    print()
    
    while True:
        try:
            user_input = input("🎨 Enter starting text (or command): ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'help':
                print("Commands: quit/exit, help, params")
                print("Or just type Chinese characters to start a poem")
                continue
            elif user_input.lower() == 'params':
                print(f"Current parameters: max_tokens=100, temperature=0.8, top_k=50")
                continue
            elif not user_input:
                continue
            
            # Check if all characters are in vocabulary
            try:
                for char in user_input:
                    if char not in stoi:
                        raise ValueError(f"Character '{char}' not in vocabulary")
            except ValueError as e:
                print(f"Error: {e}")
                continue
            
            print(f"🎯 Generating poem starting with: '{user_input}'")
            print("⏳ Please wait...")
            
            poem = generate_poem(model, itos, stoi, user_input)
            
            print("\n📜 Generated Poem:")
            print("=" * 30)
            print(poem)
            print("=" * 30)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate Tang poems with GPT model')
    parser.add_argument('--start', type=str, default="春", help='Starting text for poem generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    model, itos, stoi = load_model(
        'gpt_tang_medium_model.pt', 
        'tang_medium_meta.pkl'
    )
    
    if args.interactive:
        interactive_mode(model, itos, stoi)
    else:
        poem = generate_poem(
            model, itos, stoi, 
            args.start, 
            args.max_tokens, 
            args.temperature, 
            args.top_k
        )
        
        print("\n📜 Generated Poem:")
        print("=" * 30)
        print(poem)
        print("=" * 30)

if __name__ == "__main__":
    main() 