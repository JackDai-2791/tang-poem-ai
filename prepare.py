# prepare.py

# Read the data
with open("poems.txt", encoding='utf-8') as f:
    text = f.read()

# Build the vocabulary (unique characters)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters: {chars}")
print(f"Vocab size: {vocab_size}")

# Mapping from characters to integers and back
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Encode and decode helpers
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode entire dataset
data = encode(text)
print(f"Encoded data: {data[:100]}...")

# Save metadata for later
import pickle
with open('meta.pkl', 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos}, f)

# Save encoded data to train.bin
import torch
torch.save(torch.tensor(data, dtype=torch.long), 'train.bin')

