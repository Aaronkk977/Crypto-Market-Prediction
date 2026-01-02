#!/usr/bin/env python
"""
Train Ridge Regression model.

Usage:
    python scripts/train_ridge.py
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import train_ridge_model, scale_features, evaluate_model
from myproj.utils import save_model, save_scaler, save_config

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Set up paths
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / 'configs' / 'ridge.yaml'
    
    # Load config
    if config_path.exists():
        print(f"Loading config from {config_path}...")
        config = load_config(config_path)
    else:
        print("No config file found, using defaults...")
        config = {
            'data': {'train_path': 'data/train_fe.parquet', 'target_col': 'label'},
            'split': {'method': 'random', 'test_size': 0.2, 'random_state': 42},
            'model': {'alpha': 1.0},
            'features': {'scale_features': True},
            'training': {
                'model_dir': 'artifacts/models',
                'scaler_dir': 'artifacts/scalers'
            }
        }
    
    print("\n" + "="*60)
    print("RIDGE REGRESSION TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    data_path = project_dir / config['data']['train_path']
    df = load_data(data_path)
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        df,
        split_method=config['split']['method'],
        test_size=config['split']['test_size'],
        random_state=config['split'].get('random_state', 42),
        target_col=config['data']['target_col']
    )
    
    # Scale features
    if config['features'].get('scale_features', True):
        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    else:
        X_train_scaled, X_val_scaled = X_train, X_val
        scaler = None
    
    # Train model
    model = train_ridge_model(
        X_train_scaled, 
        y_train, 
        alpha=config['model'].get('alpha', 1.0)
    )
    
    # Evaluate model
    results = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Save artifacts
    if config['training'].get('save_model', True):
        model_dir = project_dir / config['training']['model_dir']
        model_path = model_dir / 'ridge_model.pkl'
        save_model(model, model_path)
    
    if config['training'].get('save_scaler', True) and scaler is not None:
        scaler_dir = project_dir / config['training']['scaler_dir']
        scaler_path = scaler_dir / 'scaler.pkl'
        save_scaler(scaler, scaler_path)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
