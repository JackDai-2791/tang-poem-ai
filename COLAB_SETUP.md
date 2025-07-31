# 🚀 Tang Poem AI - Google Colab Setup Guide

This guide will help you set up and train the Tang poem AI model on Google Colab with GPU acceleration.

## 📋 Prerequisites

1. **Google Colab Account** - Free GPU access
2. **Dataset Files** - You'll need to upload the prepared dataset files
3. **Model Files** - All the training and generation scripts

## 🎯 Quick Start

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → GPU

### Step 2: Upload Dataset Files
You need to upload these files to Colab:
- `tang_medium_train.bin` (Medium dataset: 2,500 poems)
- `tang_medium_meta.pkl` (Medium dataset metadata)
- `tang_large_train.bin` (Large dataset: 5,000 poems) 
- `tang_large_meta.pkl` (Large dataset metadata)

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
    "train_gpt_medium.py", 
    "train_gpt_large.py",
    "generate_gpt_medium.py",
    "generate_gpt_large.py"
]

base_url = "https://raw.githubusercontent.com/your-username/tang-poem-ai/main/"

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

### Step 5: Train the Model

#### Option A: Train Medium Model (Recommended for first run)
```python
# Train on medium dataset (2,500 poems)
!python train_gpt_medium.py
```

#### Option B: Train Large Model (For better results)
```python
# Train on large dataset (5,000 poems)
!python train_gpt_large.py
```

### Step 6: Generate Poems
```python
# Generate poems with the trained model
!python generate_gpt_medium.py --start "春"
# or
!python generate_gpt_large.py --start "春"

# Interactive mode
!python generate_gpt_medium.py --interactive
```

## 📊 Model Configurations

### Medium Model (Recommended)
- **Dataset**: 2,500 poems (179K tokens)
- **Model Size**: 24M parameters
- **Training Time**: ~30-60 minutes on Colab GPU
- **Architecture**: 6 layers, 512 embeddings, 8 heads

### Large Model (Best Quality)
- **Dataset**: 5,000 poems (336K tokens)  
- **Model Size**: 85M parameters
- **Training Time**: ~2-4 hours on Colab GPU
- **Architecture**: 12 layers, 768 embeddings, 12 heads

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
Step 0: train loss 8.62
Step 100: train loss 6.45
Step 200: train loss 5.89
Step 500: train loss 4.23
Step 1000: train loss 3.45
```

### Sample Generated Poems
```
春，花開滿院香。
東風吹綠柳，細雨潤青草。
燕子歸來時，蝴蝶舞翩翩。
```

## 🎮 Interactive Usage

### Command Line Generation
```bash
# Generate a poem starting with "春"
python generate_gpt_medium.py --start "春"

# Generate with custom parameters
python generate_gpt_medium.py --start "月" --max_tokens 150 --temperature 0.7

# Interactive mode
python generate_gpt_medium.py --interactive
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
files.download('gpt_tang_medium_model.pt')
files.download('gpt_tang_large_model.pt')
```

### Load Model Locally
```python
# Load the trained model on your local machine
import torch
from gpt_model_improved import GPTModel

model = GPTModel(vocab_size=5106, n_embd=512, n_head=8, n_layer=6, block_size=256)
model.load_state_dict(torch.load('gpt_tang_medium_model.pt'))
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in the config
   - Use medium model instead of large
   - Restart Colab runtime

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
├── train_gpt_medium.py        # Training script for medium dataset
├── train_gpt_large.py         # Training script for large dataset
├── generate_gpt_medium.py     # Generation script for medium model
├── generate_gpt_large.py      # Generation script for large model
├── tang_medium_train.bin      # Medium dataset (upload to Colab)
├── tang_medium_meta.pkl       # Medium metadata (upload to Colab)
├── tang_large_train.bin       # Large dataset (upload to Colab)
└── tang_large_meta.pkl        # Large metadata (upload to Colab)
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

The improved GPT model with pre-layer normalization should train much faster and produce better quality Tang poems compared to the basic version. 