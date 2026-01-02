"""
Utility functions for the project.
"""

import joblib
from pathlib import Path
import json
from typing import Any, Dict

def save_model(model, filepath: Path):
    """Save model to disk using joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: Path) -> Any:
    """Load model from disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def save_scaler(scaler, filepath: Path):
    """Save scaler to disk using joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath: Path) -> Any:
    """Load scaler from disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found: {filepath}")
    scaler = joblib.load(filepath)
    print(f"Scaler loaded from {filepath}")
    return scaler

def save_config(config: Dict, filepath: Path):
    """Save configuration to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {filepath}")

def load_config(filepath: Path) -> Dict:
    """Load configuration from JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Config loaded from {filepath}")
    return config
