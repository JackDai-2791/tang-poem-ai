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

def encode(text):
    return [stoi[c] for c in text]

# Sampling
def sample_model(starting_text=None, max_new_tokens=100):
    if starting_text:
        context = torch.tensor([[stoi[c] for c in starting_text]], dtype=torch.long)
    else:
        context = torch.zeros((1, 1), dtype=torch.long)  # Start from '\n' or 0

    generated = model.generate(context, max_new_tokens)[0].tolist()
    return decode(generated)

# Example usage
if __name__ == "__main__":
    print("🎭 Tang Poem AI - Quick Sample")
    print("=" * 40)
    
    # Sample with different starting characters
    starting_chars = ['春', '月', '花', '夜', '风', '雨', '明', '床', '举', '低', '思', '望']
    
    print("\n📝 Generating samples with different starting characters:")
    for char in starting_chars[:6]:  # Show first 6 for brevity
        result = sample_model(char, 50)
        print(f"\n🌸 '{char}': {result}")
    
    print("\n" + "=" * 40)
    print("🎲 Random sample (no starting character):")
    random_result = sample_model(None, 50)
    print(f"\n{random_result}")
    
    print("\n" + "=" * 40)
    print("💡 Try running 'python sample_model.py' for interactive sampling!")
    print("💡 Try running 'python batch_sample.py' for batch generation!")
