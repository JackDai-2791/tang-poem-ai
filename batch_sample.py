import torch
from train import BigramLanguageModel
import pickle

# Load metadata (char <-> int mappings)
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']

# Vocab size must match
vocab_size = len(stoi)

# Rebuild model and load weights
model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Helpers
def decode(indices):
    return ''.join([itos[i] for i in indices])

def sample_model(starting_text=None, max_new_tokens=100):
    if starting_text:
        context = torch.tensor([[stoi[c] for c in starting_text]], dtype=torch.long)
    else:
        context = torch.zeros((1, 1), dtype=torch.long)  # Start from '\n' or 0

    generated = model.generate(context, max_new_tokens)[0].tolist()
    return decode(generated)

def batch_sample():
    print("🎭 Tang Poem AI - Batch Sampling")
    print("=" * 50)
    
    # Common starting characters for Tang poetry (only those in vocabulary)
    starting_chars = ['春', '月', '花', '夜', '风', '雨', '明', '床', '举', '低', '思', '望']
    
    print(f"\nGenerating poems with {len(starting_chars)} different starting characters...")
    print("Each poem will be 50 characters long.\n")
    
    for i, char in enumerate(starting_chars, 1):
        result = sample_model(char, 50)
        print(f"{i:2d}. 🌸 '{char}': {result}")
        print()
    
    print("\n" + "=" * 50)
    print("🎲 Random poems (no starting character):")
    print()
    
    for i in range(3):
        result = sample_model(None, 50)
        print(f"{i+1}. 🎲 Random: {result}")
        print()

if __name__ == "__main__":
    batch_sample() 