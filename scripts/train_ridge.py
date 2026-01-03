#!/usr/bin/env python
"""
Train Ridge Regression model with K-Fold CV (default) or single split.

Usage:
    python scripts/train_ridge.py                    # 5-Fold CV (default)
    python scripts/train_ridge.py --cv 10            # 10-Fold CV
    python scripts/train_ridge.py --single-split     # Single random split
    python scripts/train_ridge.py --config configs/ridge.yaml
"""

import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import train_ridge_cv, scale_features
from myproj.metrics import calculate_metrics
from myproj.utils import save_model, save_scaler

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_with_kfold(df, n_folds, config, project_dir):
    """Train with K-Fold Cross-Validation."""
    print("\n" + "="*60)
    print(f"RIDGE REGRESSION WITH {n_folds}-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Prepare K-Fold splits
    fold_generator = split_data(
        df, 
        split_method='kfold', 
        n_splits=n_folds, 
        random_state=config['split'].get('random_state', 42),
        target_col=config['data']['target_col']
    )
    
    # Store results for each fold
    fold_results = []
    fold_models = []
    fold_scalers = []
    
    # Train on each fold
    for fold_idx, X_train, X_val, y_train, y_val in fold_generator:
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold_idx}/{n_folds}")
        print(f"{'='*60}")
        
        # Scale features
        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
        
        # Train model with CV for alpha selection
        model = train_ridge_cv(
            X_train_scaled, y_train,
            alphas=config['model'].get('alphas', [0.01, 0.1, 1.0, 10.0, 100.0]),
            cv=3,
            scoring='pearson'
        )
        
        # Evaluate
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        train_metrics = calculate_metrics(y_train, y_train_pred, prefix=f"Fold {fold_idx} Train")
        val_metrics = calculate_metrics(y_val, y_val_pred, prefix=f"Fold {fold_idx} Val")
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'best_alpha': float(model.alpha_),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })
        
        fold_models.append(model)
        fold_scalers.append(scaler)
    
    # Calculate average metrics
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    metrics_summary = {
        'train_r2': [r['train_metrics']['r2'] for r in fold_results],
        'train_pearson': [r['train_metrics']['pearson'] for r in fold_results],
        'val_r2': [r['val_metrics']['r2'] for r in fold_results],
        'val_pearson': [r['val_metrics']['pearson'] for r in fold_results],
        'val_rmse': [r['val_metrics']['rmse'] for r in fold_results],
        'val_mae': [r['val_metrics']['mae'] for r in fold_results]
    }
    
    # Print summary
    print("\nPer-Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: "
              f"Val R²={result['val_metrics']['r2']:.6f}, "
              f"Val Pearson={result['val_metrics']['pearson']:.6f}, "
              f"Val RMSE={result['val_metrics']['rmse']:.6f}, "
              f"Alpha={result['best_alpha']}")
    
    print(f"\nAverage Metrics ({n_folds}-Fold CV):")
    print(f"  Train R²:      {np.mean(metrics_summary['train_r2']):.6f} ± {np.std(metrics_summary['train_r2']):.6f}")
    print(f"  Train Pearson: {np.mean(metrics_summary['train_pearson']):.6f} ± {np.std(metrics_summary['train_pearson']):.6f}")
    print(f"  Val R²:        {np.mean(metrics_summary['val_r2']):.6f} ± {np.std(metrics_summary['val_r2']):.6f}")
    print(f"  Val Pearson:   {np.mean(metrics_summary['val_pearson']):.6f} ± {np.std(metrics_summary['val_pearson']):.6f}")
    print(f"  Val RMSE:      {np.mean(metrics_summary['val_rmse']):.6f} ± {np.std(metrics_summary['val_rmse']):.6f}")
    print(f"  Val MAE:       {np.mean(metrics_summary['val_mae']):.6f} ± {np.std(metrics_summary['val_mae']):.6f}")
    
    # Save results
    results_dir = project_dir / 'results' / 'ridge_cv'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with open(results_dir / 'cv_results.json', 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    summary = pd.DataFrame([{
        'Fold': r['fold'],
        'Best Alpha': r['best_alpha'],
        'Train R²': r['train_metrics']['r2'],
        'Train Pearson': r['train_metrics']['pearson'],
        'Val R²': r['val_metrics']['r2'],
        'Val Pearson': r['val_metrics']['pearson'],
        'Val RMSE': r['val_metrics']['rmse'],
        'Val MAE': r['val_metrics']['mae']
    } for r in fold_results])
    
    avg_row = pd.DataFrame([{
        'Fold': 'Average',
        'Best Alpha': '-',
        'Train R²': np.mean(metrics_summary['train_r2']),
        'Train Pearson': np.mean(metrics_summary['train_pearson']),
        'Val R²': np.mean(metrics_summary['val_r2']),
        'Val Pearson': np.mean(metrics_summary['val_pearson']),
        'Val RMSE': np.mean(metrics_summary['val_rmse']),
        'Val MAE': np.mean(metrics_summary['val_mae'])
    }])
    
    summary = pd.concat([summary, avg_row], ignore_index=True)
    summary.to_csv(results_dir / 'cv_summary.csv', index=False)
    
    print(f"\nResults saved to {results_dir}/")
    
    # Save best fold model
    best_fold_idx = np.argmax(metrics_summary['val_pearson'])
    best_model = fold_models[best_fold_idx]
    best_scaler = fold_scalers[best_fold_idx]
    
    model_dir = project_dir / config['training']['model_dir']
    scaler_dir = project_dir / config['training']['scaler_dir']
    model_dir.mkdir(exist_ok=True, parents=True)
    scaler_dir.mkdir(exist_ok=True, parents=True)
    
    save_model(best_model, model_dir / 'ridge_model.pkl')
    save_scaler(best_scaler, scaler_dir / 'scaler.pkl')
    
    print(f"\nBest fold: {best_fold_idx + 1} (Val Pearson: {metrics_summary['val_pearson'][best_fold_idx]:.6f})")
    print(f"Best model saved to {model_dir}/ridge_model.pkl")

def train_with_single_split(df, config, project_dir):
    """Train with single train/validation split."""
    print("\n" + "="*60)
    print("RIDGE REGRESSION WITH SINGLE SPLIT")
    print("="*60)
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        df,
        split_method=config['split']['method'],
        test_size=config['split']['test_size'],
        random_state=config['split'].get('random_state', 42),
        target_col=config['data']['target_col']
    )
    
    # Scale features
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    
    # Train model with CV for alpha selection
    model = train_ridge_cv(
        X_train_scaled, y_train,
        alphas=config['model'].get('alphas', [0.01, 0.1, 1.0, 10.0, 100.0]),
        cv=3,
        scoring='pearson'
    )
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    train_metrics = calculate_metrics(y_train, y_train_pred, prefix="Train")
    val_metrics = calculate_metrics(y_val, y_val_pred, prefix="Validation")
    
    # Save model and scaler
    model_dir = project_dir / config['training']['model_dir']
    scaler_dir = project_dir / config['training']['scaler_dir']
    model_dir.mkdir(exist_ok=True, parents=True)
    scaler_dir.mkdir(exist_ok=True, parents=True)
    
    save_model(model, model_dir / 'ridge_model.pkl')
    save_scaler(scaler, scaler_dir / 'scaler.pkl')
    
    print(f"\nModel saved to {model_dir}/ridge_model.pkl")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Ridge Regression model')
    parser.add_argument('--config', type=str, default='configs/ridge.yaml',
                        help='Path to config file (default: configs/ridge.yaml)')
    parser.add_argument('--single-split', action='store_true',
                        help='Use single train/val split instead of K-Fold CV')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of folds for K-Fold CV (default: 5)')
    args = parser.parse_args()
    
    # Set up paths
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / args.config
    
    # Load config
    if config_path.exists():
        print(f"Loading config from {config_path}...")
        config = load_config(config_path)
    else:
        print("No config file found, using defaults...")
        config = {
            'data': {'train_path': 'data/train_fe.parquet', 'target_col': 'label'},
            'split': {'method': 'random', 'test_size': 0.2, 'random_state': 42},
            'model': {'alphas': [0.01, 0.1, 1.0, 10.0, 100.0]},
            'training': {
                'model_dir': 'artifacts/models',
                'scaler_dir': 'artifacts/scalers'
            }
        }
    
    # Load data
    data_path = project_dir / config['data']['train_path']
    df = load_data(data_path)
    
    # Train based on mode
    if args.single_split:
        train_with_single_split(df, config, project_dir)
    else:
        train_with_kfold(df, args.cv, config, project_dir)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
