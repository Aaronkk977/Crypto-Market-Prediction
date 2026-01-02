#!/bin/bash

# Crypto Market Prediction Setup Script
# This script sets up the environment and downloads the competition data

set -e  # Exit on error

echo "========================================"
echo "Crypto Market Prediction Setup"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if kaggle is configured
if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "Warning: Kaggle API credentials not found"
    echo "Please set up your Kaggle API credentials using one of these methods:"
    echo "  1. Environment variable: export KAGGLE_API_TOKEN='YOUR_API_TOKEN'"
    echo "  2. Config file: Place kaggle.json at ~/.kaggle/kaggle.json"
    echo "Visit: https://www.kaggle.com/settings to create an API token"
fi

echo "Step 1: Creating conda environment 'crypto'..."
if conda env list | grep -q "^crypto "; then
    echo "Environment 'crypto' already exists. Skipping creation."
else
    conda create -n crypto python=3.8 -y
    echo "Environment created successfully."
fi

echo ""
echo "Step 2: Installing Python dependencies..."
# Activate conda environment and install requirements
eval "$(conda shell.bash hook)"
conda activate crypto
pip install -r requirement.txt
echo "Dependencies installed successfully."

echo ""
echo "Step 3: Creating data directory..."
mkdir -p ./data
echo "Data directory ready."

echo ""
echo "Step 4: Downloading dataset from Kaggle..."
if [ -f ~/.kaggle/kaggle.json ] || [ -n "$KAGGLE_API_TOKEN" ]; then
    # Download the competition data
    kaggle competitions download -c drw-crypto-market-prediction
    
    echo "Extracting dataset..."
    unzip -o drw-crypto-market-prediction.zip -d ./data/
    
    # Clean up zip file
    rm drw-crypto-market-prediction.zip
    
    echo "Dataset downloaded and extracted successfully."
else
    echo "Skipping dataset download (Kaggle API not configured)"
    echo "Please set up Kaggle credentials and run the download manually:"
    echo "  export KAGGLE_API_TOKEN='YOUR_API_TOKEN'"
    echo "  kaggle competitions download -c drw-crypto-market-prediction"
    echo "Or download manually from: https://www.kaggle.com/competitions/drw-crypto-market-prediction/data"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate crypto"
echo ""
echo "To explore the data, run:"
echo "  python read_parquet.py"
echo ""
echo "To start training, run:"
echo "  python main.py"
echo ""