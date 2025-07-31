#!/usr/bin/env python3
"""
Setup script for Google Colab training.
This script will download all necessary files and prepare the environment.
"""

import os
import requests
import zipfile
from google.colab import files
import torch

def download_file_from_github(url, filename):
    """Download a file from GitHub raw content."""
    print(f"Downloading {filename}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"✅ Downloaded {filename}")
    else:
        print(f"❌ Failed to download {filename}")

def setup_colab_environment():
    """Set up the Colab environment for Tang poem training."""
    
    print("🚀 Setting up Tang Poem AI training environment for Colab...")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠️  GPU not available, using CPU")
        device = 'cpu'
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Download the model files
    base_url = "https://raw.githubusercontent.com/your-username/tang-poem-ai/main/"
    
    files_to_download = [
        "gpt_model_improved.py",
        "train_gpt_medium.py", 
        "train_gpt_large.py",
        "generate_gpt_medium.py",
        "generate_gpt_large.py"
    ]
    
    for file in files_to_download:
        download_file_from_github(base_url + file, file)
    
    print("\n📁 Files downloaded successfully!")
    print("📋 Next steps:")
    print("1. Upload your dataset files (tang_medium_train.bin, tang_medium_meta.pkl)")
    print("2. Run the training script")
    print("3. Download the trained model")
    
    return device

if __name__ == "__main__":
    device = setup_colab_environment()
    print(f"\n🎯 Ready to train on {device.upper()}") 