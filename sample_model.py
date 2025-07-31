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

# Sampling function
def sample_model(starting_text=None, max_new_tokens=100, temperature=1.0):
    if starting_text:
        context = torch.tensor([[stoi[c] for c in starting_text]], dtype=torch.long)
    else:
        context = torch.zeros((1, 1), dtype=torch.long)  # Start from '\n' or 0

    generated = model.generate(context, max_new_tokens)[0].tolist()
    return decode(generated)

def interactive_sample():
    print("🎭 Tang Poem AI Model Sampler")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Generate with custom starting text")
        print("2. Generate random poem")
        print("3. Try different starting characters")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            start_text = input("Enter starting text (e.g., '春', '月', '花'): ").strip()
            if start_text:
                try:
                    max_tokens = int(input("Enter max tokens (default 100): ") or "100")
                    result = sample_model(start_text, max_tokens)
                    print(f"\n🎯 Generated poem starting with '{start_text}':")
                    print("-" * 40)
                    print(result)
                except ValueError:
                    print("Invalid input. Using default values.")
                    result = sample_model(start_text, 100)
                    print(f"\n🎯 Generated poem starting with '{start_text}':")
                    print("-" * 40)
                    print(result)
            else:
                print("Please enter some starting text.")
                
        elif choice == '2':
            try:
                max_tokens = int(input("Enter max tokens (default 100): ") or "100")
                result = sample_model(None, max_tokens)
                print(f"\n🎲 Random generated poem:")
                print("-" * 40)
                print(result)
            except ValueError:
                print("Invalid input. Using default values.")
                result = sample_model(None, 100)
                print(f"\n🎲 Random generated poem:")
                print("-" * 40)
                print(result)
                
        elif choice == '3':
            common_chars = ['春', '月', '花', '夜', '山', '水', '风', '雨', '雪', '秋']
            print(f"\nCommon starting characters: {', '.join(common_chars)}")
            for char in common_chars:
                result = sample_model(char, 50)
                print(f"\n🌺 '{char}': {result}")
                
        elif choice == '4':
            print("Goodbye! 👋")
            break
            
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    interactive_sample() 