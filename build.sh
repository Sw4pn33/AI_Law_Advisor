#!/bin/bash
set -e

echo "[Build] Installing CPU-only PyTorch first (skip CUDA)..."
pip install torch==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

echo "[Build] Installing Python dependencies..."
pip install -r requirements.txt

echo "[Build] Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "[Build] Build complete!"
