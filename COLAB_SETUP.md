# 🚀 Tang Poem AI - Google Colab Setup Guide (Large Dataset)

This guide will help you set up and train the Tang poem AI model on Google Colab with GPU acceleration, optimized for the large dataset (5,000 poems).

## 📋 Prerequisites

1. **Google Colab Account** - Free GPU access
2. **Dataset Files** - You'll need to upload the prepared large dataset files
3. **Model Files** - All the training and generation scripts

## 🎯 Quick Start

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → GPU

### Step 2: Upload Dataset Files
You need to upload these files to Colab:
- `tang_large_train.bin` (Large dataset: 5,000 poems, 2.7MB)
- `tang_large_meta.pkl` (Large dataset metadata, 94KB)

### Step 3: Install Dependencies
```python
# Install PyTorch (Colab usually has it pre-installed)
!pip install torch torchvision torchaudio
```

### Step 4: Download Model Files
```python
# Download all necessary files from GitHub
import requests

files_to_download = [
    "gpt_model_improved.py",
    "train_gpt_large.py",
    "generate_gpt_large.py"
]

base_url = "https://raw.githubusercontent.com/JackDai-2791/tang-poem-ai/main/"

for file in files_to_download:
    print(f"Downloading {file}...")
    response = requests.get(base_url + file)
    if response.status_code == 200:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"✅ Downloaded {file}")
    else:
        print(f"❌ Failed to download {file}")
```

### Step 5: Train the Large Model
```python
# Train on large dataset (5,000 poems)
!python train_gpt_large.py
```

### Step 6: Generate Poems
```python
# Generate poems with the trained model
!python generate_gpt_large.py --start "春"

# Interactive mode
!python generate_gpt_large.py --interactive
```

## 📊 Model Configuration

### Large Model (Best Quality)
- **Dataset**: 5,000 poems (336K tokens)
- **Model Size**: 85M parameters
- **Training Time**: ~2-4 hours on Colab GPU
- **Architecture**: 12 layers, 768 embeddings, 12 heads
- **Vocabulary**: 5,618 unique Chinese characters

## 🔧 Advanced Features

### Improved Architecture
- ✅ **Pre-layer normalization** for better training stability
- ✅ **Multi-head self-attention** with causal masking
- ✅ **Positional encoding** for sequence awareness
- ✅ **Residual connections** throughout the network
- ✅ **Gradient clipping** to prevent exploding gradients
- ✅ **Learning rate scheduling** for optimal convergence

### Training Optimizations
- **AdamW optimizer** with weight decay
- **Cosine annealing** learning rate schedule
- **Gradient clipping** at 1.0
- **Dropout** for regularization
- **Large batch sizes** for GPU efficiency

## 📈 Expected Results

### Training Progress
```
Step 0: train loss 8.85
Step 500: train loss 6.23
Step 1000: train loss 4.23
Step 2000: train loss 3.45
Step 5000: train loss 2.98
```

### Sample Generated Poems
```
春，花開滿院香。
東風吹綠柳，細雨潤青草。
燕子歸來時，蝴蝶舞翩翩。

月，清輝灑人間。
銀河倒影水，玉露滴花前。
夜深人靜處，思緒萬千般。
```

## 🎮 Interactive Usage

### Command Line Generation
```bash
# Generate a poem starting with "春"
python generate_gpt_large.py --start "春"

# Generate with custom parameters
python generate_gpt_large.py --start "月" --max_tokens 150 --temperature 0.7

# Interactive mode
python generate_gpt_large.py --interactive
```

### Interactive Commands
- Type Chinese characters to start a poem
- `quit` or `exit` - Exit the program
- `help` - Show help
- `params` - Show current parameters

## 💾 Model Management

### Download Trained Model
After training, download the model file:
```python
from google.colab import files
files.download('gpt_tang_large_model.pt')
```

### Load Model Locally
```python
# Load the trained model on your local machine
import torch
from gpt_model_improved import GPTModel

model = GPTModel(vocab_size=5618, n_embd=768, n_head=12, n_layer=12, block_size=512)
model.load_state_dict(torch.load('gpt_tang_large_model.pt'))
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in the config (change from 64 to 32)
   - Restart Colab runtime
   - Use smaller model if needed

2. **Slow Training**
   - Ensure GPU is enabled
   - Check if CUDA is available: `torch.cuda.is_available()`

3. **Model Not Converging**
   - Increase training iterations
   - Adjust learning rate
   - Check dataset quality

### Performance Tips

1. **Use GPU Runtime**: Always enable GPU in Colab
2. **Monitor Memory**: Watch for OOM errors
3. **Save Checkpoints**: Download models periodically
4. **Use Mixed Precision**: For even faster training (optional)

## 📁 File Structure

```
tang-poem-ai/
├── gpt_model_improved.py      # Improved GPT model architecture
├── train_gpt_large.py         # Training script for large dataset
├── generate_gpt_large.py      # Generation script for large model
├── create_large_dataset.py    # Dataset creation script
├── prepare_large_dataset.py   # Dataset preparation script
├── extract_tang_poems.py      # Extract from 全唐诗 repository
├── tang_large_train.bin       # Large dataset (upload to Colab)
├── tang_large_meta.pkl        # Large metadata (upload to Colab)
├── COLAB_SETUP.md             # This guide
├── colab_setup.py             # Colab environment setup
└── tang_poem_ai_colab.ipynb   # Colab notebook template
```

## 🎉 Success Metrics

- **Loss < 4.0**: Good training progress
- **Loss < 3.0**: Excellent model quality
- **Coherent Chinese text**: Model understands language
- **Proper poem structure**: Rhyming and formatting

## 🔮 Next Steps

1. **Train on full dataset**: Use all 100K+ poems
2. **Experiment with architecture**: Try different model sizes
3. **Add poem constraints**: Enforce traditional poem structure
4. **Fine-tune parameters**: Optimize temperature, top-k, etc.
5. **Deploy model**: Create web interface for poem generation

---

**Happy Training! 🚀**

The improved GPT model with pre-layer normalization should train efficiently on Colab GPU and produce high-quality Tang poems with the large dataset. 