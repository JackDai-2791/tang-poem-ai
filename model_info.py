import torch
import pickle
from train import BigramLanguageModel

def inspect_model():
    """Inspect model information and vocabulary."""
    print("🔍 Tang Poem AI - Model Inspector")
    print("=" * 50)
    
    # Load metadata
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    
    print(f"\n📊 Model Information:")
    print(f"   Vocabulary size: {len(stoi)} characters")
    print(f"   Available characters: {list(stoi.keys())}")
    
    # Load model
    model = BigramLanguageModel(len(stoi))
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    
    print(f"\n🧠 Model Architecture:")
    print(f"   Type: Bigram Language Model")
    print(f"   Embedding size: {model.token_embedding_table.embedding_dim}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n📝 Character Mappings:")
    for i, char in enumerate(itos):
        print(f"   {i:2d}: '{char}'")
    
    print(f"\n🎯 Most Common Starting Characters:")
    common_chars = ['春', '月', '花', '夜', '风', '雨', '明', '床', '举', '低', '思', '望']
    for char in common_chars:
        if char in stoi:
            print(f"   '{char}' (ID: {stoi[char]})")
    
    print(f"\n💡 Usage Tips:")
    print(f"   - Use 'python generate.py' for quick sampling")
    print(f"   - Use 'python sample_model.py' for interactive mode")
    print(f"   - Use 'python batch_sample.py' for batch generation")

if __name__ == "__main__":
    inspect_model() 