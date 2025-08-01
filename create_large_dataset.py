#!/usr/bin/env python3
"""
Create a large subset of Tang poems for Colab training.
"""

import random

def create_large_dataset(input_file="tang_poems_full.txt", output_file="tang_poems_large.txt", num_poems=5000):
    """Create a large subset of poems for Colab training."""
    
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into individual poems (separated by double newlines)
    poems = content.split('\n\n')
    poems = [poem.strip() for poem in poems if poem.strip()]
    
    print(f"Total poems found: {len(poems)}")
    
    # Randomly sample poems
    if len(poems) > num_poems:
        selected_poems = random.sample(poems, num_poems)
    else:
        selected_poems = poems
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(selected_poems))
    
    print(f"Created large dataset with {len(selected_poems)} poems")
    print(f"Saved to {output_file}")
    
    # Show some statistics
    total_chars = sum(len(poem) for poem in selected_poems)
    print(f"Total characters: {total_chars:,}")
    print(f"Average poem length: {total_chars // len(selected_poems):,} characters")
    
    # Show a few sample poems
    print("\nSample poems:")
    print("=" * 50)
    for i, poem in enumerate(selected_poems[:3]):
        print(f"Poem {i+1}:")
        print(poem)
        print("-" * 30)

if __name__ == "__main__":
    create_large_dataset() 