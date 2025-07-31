#!/usr/bin/env python3
"""
Extract Tang poems from the chinese-poetry repository and convert to training format.
"""

import json
import os
import glob
from pathlib import Path

def extract_poems_from_json(json_file_path):
    """Extract poems from a single JSON file."""
    poems = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for poem in data:
            if 'paragraphs' in poem and poem['paragraphs']:
                # Join all paragraphs with newlines
                poem_text = '\n'.join(poem['paragraphs'])
                poems.append(poem_text)
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
    
    return poems

def extract_all_tang_poems():
    """Extract all Tang poems from the chinese-poetry repository."""
    all_poems = []
    
    # Path to the chinese-poetry repository
    poetry_dir = Path("chinese-poetry")
    
    # Extract from 全唐诗 directory
    tang_dir = poetry_dir / "全唐诗"
    if tang_dir.exists():
        print("Extracting from 全唐诗 directory...")
        json_files = glob.glob(str(tang_dir / "poet.tang.*.json"))
        
        for json_file in json_files:
            print(f"Processing {json_file}...")
            poems = extract_poems_from_json(json_file)
            all_poems.extend(poems)
            print(f"  Found {len(poems)} poems")
    
    # Extract from 御定全唐詩 directory
    yuding_dir = poetry_dir / "御定全唐詩" / "json"
    if yuding_dir.exists():
        print("Extracting from 御定全唐詩 directory...")
        json_files = glob.glob(str(yuding_dir / "*.json"))
        
        for json_file in json_files:
            print(f"Processing {json_file}...")
            poems = extract_poems_from_json(json_file)
            all_poems.extend(poems)
            print(f"  Found {len(poems)} poems")
    
    return all_poems

def save_poems_to_file(poems, output_file="tang_poems_full.txt"):
    """Save poems to a text file in the training format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, poem in enumerate(poems):
            f.write(poem)
            f.write('\n\n')  # Add double newline between poems
    
    print(f"Saved {len(poems)} poems to {output_file}")

def main():
    """Main function to extract and save Tang poems."""
    print("Starting Tang poem extraction...")
    
    # Extract all poems
    poems = extract_all_tang_poems()
    
    print(f"\nTotal poems extracted: {len(poems)}")
    
    if poems:
        # Save to file
        save_poems_to_file(poems)
        
        # Show some statistics
        total_chars = sum(len(poem) for poem in poems)
        avg_chars = total_chars / len(poems) if poems else 0
        
        print(f"\nStatistics:")
        print(f"  Total poems: {len(poems)}")
        print(f"  Total characters: {total_chars}")
        print(f"  Average characters per poem: {avg_chars:.1f}")
        
        # Show a few sample poems
        print(f"\nSample poems:")
        for i in range(min(3, len(poems))):
            print(f"\n--- Poem {i+1} ---")
            print(poems[i])
    else:
        print("No poems found!")

if __name__ == "__main__":
    main() 