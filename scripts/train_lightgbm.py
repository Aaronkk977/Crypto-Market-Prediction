#!/usr/bin/env python
"""
Train LightGBM model using configuration file.

Usage:
    python scripts/train_lightgbm.py [--config configs/lightgbm.yaml]
"""

import sys
from pathlib import Path
import yaml
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import train_lightgbm
from myproj.metrics import calculate_metrics

try:
    import lightgbm as lgb
except ImportError:
    print("Error: LightGBM is not installed.")
    print("Install with: pip install lightgbm")
    sys.exit(1)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train LightGBM model')
    parser.add_argument('--config', type=str, default='configs/lightgbm.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / args.config
    config = load_config(config_path)
    
    print("\n" + "="*60)
    print("LIGHTGBM MODEL TRAINING")
    print("="*60)
    print(f"Config: {config_path}")
    
    # Load data
    data_dir = project_dir / 'data'
    train_path = data_dir / Path(config['data']['train_path']).name
    df = load_data(train_path)
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        df,
        split_method=config['training']['split_method'],
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples")
    
    # Train model
    model = train_lightgbm(
        X_train, y_train, X_val, y_val,
        params=config['model'],
        num_boost_round=config['training']['num_boost_round'],
        early_stopping_rounds=config['training']['early_stopping_rounds']
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    train_metrics = calculate_metrics(y_train, y_train_pred, prefix="Train")
    val_metrics = calculate_metrics(y_val, y_val_pred, prefix="Validation")
    
    # Save model
    model_dir = project_dir / config['output']['model_dir']
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / config['output']['model_name']
    
    print(f"\nSaving model to {model_path}...")
    model.save_model(str(model_path))
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
