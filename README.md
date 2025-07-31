# 🚀 Tang Poem AI - GPT-Style Transformer (Large Dataset)

A complete implementation of a GPT-style transformer model for generating Tang dynasty poems in Chinese, optimized for large-scale training on Google Colab with GPU acceleration.

## 🎯 Project Overview

This project implements a state-of-the-art language model for Chinese Tang poetry generation, featuring:

- **GPT-style transformer architecture** with pre-layer normalization
- **Multi-head self-attention** with causal masking
- **Positional encoding** for sequence awareness
- **Residual connections** throughout the network
- **Large dataset training** (5,000 poems, 336K tokens)
- **GPU-optimized training** for Google Colab

## 📊 Model Configuration

### Large Model (Best Quality)
- **Dataset**: 5,000 poems (336K tokens)
- **Model Size**: 85M parameters
- **Architecture**: 12 layers, 768 embeddings, 12 heads
- **Training Time**: ~2-4 hours on Colab GPU
- **Vocabulary**: 5,618 unique Chinese characters

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
│   └── gpt_model_improved.py     # Improved GPT model with pre-layer norm
│
├── 🎯 Training Scripts
│   ├── train_gpt_large.py        # Large model training
│   └── generate_gpt_large.py     # Large model generation
│
├── 📊 Dataset Preparation
│   ├── extract_tang_poems.py     # Extract from 全唐诗 repository
│   ├── create_large_dataset.py   # Create 5,000-poem subset
│   └── prepare_large_dataset.py  # Prepare large dataset
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

### Google Colab Training (Recommended)
1. **Open** [Google Colab](https://colab.research.google.com/)
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Upload dataset files**: `tang_large_train.bin`, `tang_large_meta.pkl`
4. **Download scripts** from GitHub
5. **Train model**: `!python train_gpt_large.py`
6. **Generate poems**: `!python generate_gpt_large.py --start "春"`

See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed instructions.

## 🎭 Usage Examples

### Command Line Generation
```bash
# Generate a poem starting with "春"
python3 generate_gpt_large.py --start "春"

# Generate with custom parameters
python3 generate_gpt_large.py --start "月" --max_tokens 150 --temperature 0.7

# Interactive mode
python3 generate_gpt_large.py --interactive
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
Large Model (5,000 poems):
Step 0: train loss 8.85
Step 1000: train loss 4.23
Step 2000: train loss 3.45
Step 5000: train loss 2.98
```

## 🔧 Technical Details

### Model Architecture
- **Token Embeddings**: Learnable character embeddings
- **Positional Encoding**: Sinusoidal positional embeddings
- **Transformer Blocks**: Pre-layer norm + attention + feed-forward
- **Output Projection**: Linear layer to vocabulary size

### Training Configuration
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 1e-4 (optimized for large model)
- **Batch Size**: 64 (GPU optimized)
- **Block Size**: 512 (larger context window)
- **Dropout**: 0.1 throughout

### Generation Parameters
- **Temperature**: 0.8 (controls randomness)
- **Top-k**: 50 (nucleus sampling)
- **Max Tokens**: 150 (poem length)

## 🎯 Performance Metrics

### Quality Indicators
- **Loss < 4.0**: Good training progress
- **Loss < 3.0**: Excellent model quality
- **Coherent Chinese text**: Model understands language
- **Proper poem structure**: Rhyming and formatting

### Speed Benchmarks
- **Large Model**: ~2-4 hours on Colab GPU
- **Generation**: Real-time poem creation
- **Memory Usage**: ~8GB GPU memory

## 🚨 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use smaller model
2. **Slow Training**: Ensure GPU is enabled in Colab
3. **Poor Quality**: Increase training iterations

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

## 📚 Dataset Information

### Source
- **Repository**: [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- **Content**: Complete Tang dynasty poems (全唐诗)
- **Format**: Traditional Chinese characters
- **Size**: 100,706 poems total

### Processing
- **Character-level tokenization**: Each character = one token
- **Vocabulary size**: 5,618 unique characters
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

The improved GPT model with pre-layer normalization provides excellent results for Chinese Tang poetry generation, optimized for large-scale training on Google Colab with GPU acceleration. 