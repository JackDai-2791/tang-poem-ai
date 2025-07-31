# 🚀 Tang Poem AI - GPT-Style Transformer

A complete implementation of a GPT-style transformer model for generating Tang dynasty poems in Chinese. This project includes improved architecture with pre-layer normalization, multiple dataset sizes, and comprehensive training/generation scripts.

## 🎯 Project Overview

This project implements a state-of-the-art language model for Chinese Tang poetry generation, featuring:

- **GPT-style transformer architecture** with pre-layer normalization
- **Multi-head self-attention** with causal masking
- **Positional encoding** for sequence awareness
- **Residual connections** throughout the network
- **Gradual dataset scaling** (500 → 2,500 → 5,000 poems)
- **GPU-optimized training** for Google Colab

## 📊 Model Configurations

### Small Model (Testing)
- **Dataset**: 500 poems (35K tokens)
- **Model Size**: 4.8M parameters
- **Architecture**: 4 layers, 256 embeddings, 4 heads
- **Training Time**: ~10-15 minutes on CPU

### Medium Model (Recommended)
- **Dataset**: 2,500 poems (179K tokens)
- **Model Size**: 24M parameters
- **Architecture**: 6 layers, 512 embeddings, 8 heads
- **Training Time**: ~30-60 minutes on GPU

### Large Model (Best Quality)
- **Dataset**: 5,000 poems (336K tokens)
- **Model Size**: 85M parameters
- **Architecture**: 12 layers, 768 embeddings, 12 heads
- **Training Time**: ~2-4 hours on GPU

## 🏗️ Architecture Improvements

### Pre-Layer Normalization
```python
# Layer norm before attention and feed-forward
x_norm = self.ln(x)
out = self.attention(x_norm) + x  # Residual connection
```

### Multi-Head Self-Attention
- **Causal masking** prevents looking at future tokens
- **Scaled dot-product attention** with temperature scaling
- **Dropout** for regularization

### Training Optimizations
- **AdamW optimizer** with weight decay
- **Cosine annealing** learning rate schedule
- **Gradient clipping** at 1.0
- **Large batch sizes** for GPU efficiency

## 📁 File Structure

```
tang-poem-ai/
├── 📄 Model Architecture
│   ├── gpt_model.py              # Basic GPT model
│   └── gpt_model_improved.py     # Improved model with pre-layer norm
│
├── 🎯 Training Scripts
│   ├── train_gpt_small.py        # Small model training
│   ├── train_gpt_medium.py       # Medium model training
│   └── train_gpt_large.py        # Large model training
│
├── 🎭 Generation Scripts
│   ├── generate_gpt_small.py     # Small model generation
│   ├── generate_gpt_medium.py    # Medium model generation
│   └── generate_gpt_large.py     # Large model generation
│
├── 📊 Dataset Preparation
│   ├── extract_tang_poems.py     # Extract from 全唐诗 repository
│   ├── create_small_dataset.py   # Create 500-poem subset
│   ├── create_medium_dataset.py  # Create 2,500-poem subset
│   ├── create_large_dataset.py   # Create 5,000-poem subset
│   ├── prepare_small_dataset.py  # Prepare small dataset
│   ├── prepare_medium_dataset.py # Prepare medium dataset
│   └── prepare_large_dataset.py  # Prepare large dataset
│
├── 📚 Datasets
│   ├── tang_small_train.bin      # Small dataset (35K tokens)
│   ├── tang_small_meta.pkl       # Small metadata
│   ├── tang_medium_train.bin     # Medium dataset (179K tokens)
│   ├── tang_medium_meta.pkl      # Medium metadata
│   ├── tang_large_train.bin      # Large dataset (336K tokens)
│   └── tang_large_meta.pkl       # Large metadata
│
├── 🚀 Colab Setup
│   ├── COLAB_SETUP.md            # Comprehensive Colab guide
│   ├── colab_setup.py            # Colab setup script
│   └── tang_poem_ai_colab.ipynb  # Colab notebook template
│
└── 📖 Documentation
    └── README.md                 # This file
```

## 🚀 Quick Start

### Local Training (CPU - Slow)
```bash
# Train small model
python3 train_gpt_small.py

# Generate poems
python3 generate_gpt_small.py --start "春"
python3 generate_gpt_small.py --interactive
```

### Google Colab Training (GPU - Fast)
1. **Open** [Google Colab](https://colab.research.google.com/)
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Upload dataset files**: `tang_medium_train.bin`, `tang_medium_meta.pkl`
4. **Download scripts** from GitHub
5. **Train model**: `!python train_gpt_medium.py`
6. **Generate poems**: `!python generate_gpt_medium.py --start "春"`

See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed instructions.

## 🎭 Usage Examples

### Command Line Generation
```bash
# Generate a poem starting with "春"
python3 generate_gpt_medium.py --start "春"

# Generate with custom parameters
python3 generate_gpt_medium.py --start "月" --max_tokens 150 --temperature 0.7

# Interactive mode
python3 generate_gpt_medium.py --interactive
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

## 📈 Training Progress

### Expected Loss Curves
```
Small Model (500 poems):
Step 0: train loss 8.12
Step 500: train loss 5.60
Step 999: train loss 4.59

Medium Model (2,500 poems):
Step 0: train loss 8.62
Step 1000: train loss 4.23
Step 1999: train loss 3.45

Large Model (5,000 poems):
Step 0: train loss 8.85
Step 2000: train loss 3.89
Step 4999: train loss 2.98
```

## 🔧 Technical Details

### Model Architecture
- **Token Embeddings**: Learnable character embeddings
- **Positional Encoding**: Sinusoidal positional embeddings
- **Transformer Blocks**: Pre-layer norm + attention + feed-forward
- **Output Projection**: Linear layer to vocabulary size

### Training Configuration
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 3e-4 (medium), 1e-4 (large)
- **Batch Size**: 32 (medium), 64 (large)
- **Block Size**: 256 (medium), 512 (large)
- **Dropout**: 0.1 throughout

### Generation Parameters
- **Temperature**: 0.8 (controls randomness)
- **Top-k**: 50 (nucleus sampling)
- **Max Tokens**: 100-150 (poem length)

## 🎯 Performance Metrics

### Quality Indicators
- **Loss < 4.0**: Good training progress
- **Loss < 3.0**: Excellent model quality
- **Coherent Chinese text**: Model understands language
- **Proper poem structure**: Rhyming and formatting

### Speed Benchmarks
- **Small Model**: ~10-15 min on CPU
- **Medium Model**: ~30-60 min on GPU
- **Large Model**: ~2-4 hours on GPU

## 🚨 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use smaller model
2. **Slow Training**: Enable GPU in Colab
3. **Poor Quality**: Increase dataset size or training iterations

### Performance Tips
1. **Use GPU**: Always enable GPU for training
2. **Monitor Loss**: Watch for convergence
3. **Save Checkpoints**: Download models periodically
4. **Experiment**: Try different generation parameters

## 🔮 Future Improvements

### Planned Enhancements
1. **Full Dataset Training**: Use all 100K+ poems
2. **Poem Structure Constraints**: Enforce traditional formats
3. **Rhyme Detection**: Add rhyming capabilities
4. **Web Interface**: Deploy as web application
5. **Fine-tuning**: Optimize for specific styles

### Research Directions
1. **Attention Visualization**: Understand model focus
2. **Style Transfer**: Generate different poetic styles
3. **Multi-modal**: Combine with image generation
4. **Interactive Editing**: Real-time poem refinement

## 📚 Dataset Information

### Source
- **Repository**: [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- **Content**: Complete Tang dynasty poems (全唐诗)
- **Format**: Traditional Chinese characters
- **Size**: 100,706 poems total

### Processing
- **Character-level tokenization**: Each character = one token
- **Vocabulary size**: 3,218 (small) → 5,106 (medium) → 5,618 (large)
- **Encoding**: UTF-8 with proper Chinese character handling

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- **Model architecture** enhancements
- **Training optimizations**
- **Dataset processing** improvements
- **Generation quality** enhancements
- **Documentation** and examples

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **Dataset**: [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) repository
- **Architecture**: Inspired by GPT and transformer research
- **Implementation**: Built with PyTorch

---

**Happy poem generating! 🚀**

The improved GPT model with pre-layer normalization provides excellent results for Chinese Tang poetry generation, with training times optimized for both local development and cloud GPU acceleration. 