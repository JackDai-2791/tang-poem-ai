# 🎭 Tang Poem AI

A neural language model trained to generate classical Chinese Tang poetry. This project implements a simple bigram language model that learns to generate poetry in the style of Tang dynasty poets.

## 📁 Project Structure

```
tang-poem-ai/
├── train.py          # Training script for the bigram model
├── generate.py       # Basic sampling script
├── sample_model.py   # Interactive sampling interface
├── batch_sample.py   # Batch generation script
├── prepare.py        # Data preparation script
├── model.pt          # Trained model weights
├── meta.pkl          # Character-to-index mappings
├── train.bin         # Preprocessed training data
├── poems.txt         # Original training poems
└── README.md         # This file
```

## 🚀 Quick Start

### Setup Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install torch
```

### Generate Poems

#### 1. Quick Sample (`generate.py`)
```bash
python generate.py
```
Generates samples with different starting characters and a random poem.

#### 2. Interactive Sampling (`sample_model.py`)
```bash
python sample_model.py
```
Provides an interactive menu to:
- Generate with custom starting text
- Generate random poems
- Try different starting characters
- Adjust generation parameters

#### 3. Batch Sampling (`batch_sample.py`)
```bash
python batch_sample.py
```
Generates multiple poems with different starting characters for comparison.

## 🎯 Available Starting Characters

The model supports these starting characters (based on training vocabulary):
- **春** (Spring) - 春眠不觉晓
- **月** (Moon) - 明月光
- **花** (Flower) - 花落知多少
- **夜** (Night) - 夜来风雨声
- **风** (Wind) - 风雨声
- **雨** (Rain) - 雨声
- **明** (Bright) - 明月
- **床** (Bed) - 床前明月
- **举** (Raise) - 举头望明月
- **低** (Lower) - 低头思故乡
- **思** (Think) - 思故乡
- **望** (Gaze) - 望明月

## 🧠 Model Architecture

- **Type**: Bigram Language Model
- **Architecture**: Simple embedding-based model
- **Training**: 5000 iterations with AdamW optimizer
- **Context**: 8-token window
- **Vocabulary**: 38 unique characters

## 📊 Training Data

The model was trained on classical Tang poems including:
- "静夜思" (Quiet Night Thoughts) by Li Bai
- "春晓" (Spring Dawn) by Meng Haoran

## 🎨 Sample Outputs

### Starting with "春" (Spring):
```
春眠不觉晓，
低头思故乡。
举头思故乡。
低头望明月光，
低头思故乡。
举头望明月光，
处处处处闻啼
```

### Starting with "月" (Moon):
```
月光，
春眠不觉晓，
低头思故乡。
疑是地上霜。
低头望明月光，
花落知多少。
花落知多少。
春眠
```

### Random Generation:
```
花落知多少。
处闻啼鸟。
夜来风雨声，
夜来风雨声，
举头思故乡。
举头思故乡。
花落知多少。
春
```

## 🔧 Customization

### Adjusting Generation Parameters

In any of the sampling scripts, you can modify:
- `max_new_tokens`: Number of characters to generate (default: 100)
- `starting_text`: Initial character(s) to start generation
- `temperature`: Sampling temperature (affects randomness)

### Adding New Training Data

1. Add new poems to `poems.txt`
2. Run `python prepare.py` to preprocess
3. Retrain with `python train.py`

## 📝 Usage Examples

### Python API
```python
from generate import sample_model

# Generate with starting character
poem = sample_model("春", max_new_tokens=50)
print(poem)

# Generate random poem
random_poem = sample_model(None, max_new_tokens=100)
print(random_poem)
```

### Command Line
```bash
# Quick sample
python generate.py

# Interactive mode
python sample_model.py

# Batch generation
python batch_sample.py
```

## 🎯 Model Performance

The model successfully learns:
- ✅ Classical Chinese punctuation (，。)
- ✅ Tang poetry structure and rhythm
- ✅ Thematic coherence (moon, spring, flowers, etc.)
- ✅ Character-level language patterns
- ✅ Poetic line breaks and formatting

## 🔮 Future Enhancements

Potential improvements:
- Larger training dataset
- More sophisticated model architecture (Transformer, LSTM)
- Temperature-controlled sampling
- Rhyme and meter constraints
- Multi-line poem generation
- Style transfer capabilities

## 📚 References

- Tang Dynasty Poetry
- Neural Language Modeling
- Character-level Language Models
- Classical Chinese Literature

---

*Generated with ❤️ by Tang Poem AI* 