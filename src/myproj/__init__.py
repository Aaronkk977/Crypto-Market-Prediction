"""
MyProj - Crypto Market Prediction Package

A modular machine learning package for cryptocurrency market prediction.
"""

__version__ = "0.1.0"

from .data import load_data
from .features import add_depth_features
from .split import split_data, random_split, time_split
from .models import train_ridge_model, evaluate_model
from .metrics import calculate_metrics

__all__ = [
    'load_data',
    'add_depth_features',
    'split_data',
    'random_split',
    'time_split',
    'train_ridge_model',
    'evaluate_model',
    'calculate_metrics',
]
