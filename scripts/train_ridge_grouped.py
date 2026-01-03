#!/usr/bin/env python
"""
Train Ridge Regression model with GroupKFold CV to prevent label leakage.

Usage:
    python scripts/train_ridge_grouped.py                    # 5-Fold GroupKFold CV (default)
    python scripts/train_ridge_grouped.py --cv 10            # 10-Fold GroupKFold CV
    python scripts/train_ridge_grouped.py --config configs/ridge.yaml
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

def train_with_group_kfold(df, n_folds, config, project_dir):
    """Train with GroupKFold Cross-Validation (prevents label leakage)."""
    print("\n" + "="*60)
    print(f"RIDGE REGRESSION WITH {n_folds}-FOLD GROUPKFOLD CV")
    print("="*60)
    print("⚠️  Using GroupKFold to prevent temporal label leakage")
    
    # Prepare GroupKFold splits
    fold_generator = split_data(
        df, 
        split_method='group_kfold',
        group_col='time_group_id',
        n_splits=n_folds, 
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
    print("GROUPKFOLD CROSS-VALIDATION RESULTS")
    print("="*60)
    
    metrics_summary = {
        'train_r2': [r['train_metrics']['r2'] for r in fold_results],
        'train_pearson': [r['train_metrics']['pearson'] for r in fold_results],
        'val_r2': [r['val_metrics']['r2'] for r in fold_results],
        'val_pearson': [r['val_metrics']['pearson'] for r in fold_results],
        'val_rmse': [r['val_metrics']['rmse'] for r in fold_results],
        'val_mae': [r['val_metrics']['mae'] for r in fold_results]
    }
    
    for metric_name, values in metrics_summary.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name:20s}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Find best fold by validation Pearson correlation
    best_fold_idx = np.argmax(metrics_summary['val_pearson'])
    best_model = fold_models[best_fold_idx]
    best_scaler = fold_scalers[best_fold_idx]
    
    print(f"\n{'='*60}")
    print(f"BEST FOLD: {best_fold_idx + 1}")
    print(f"{'='*60}")
    print(f"Best Alpha: {fold_results[best_fold_idx]['best_alpha']}")
    print(f"Val Pearson: {fold_results[best_fold_idx]['val_metrics']['pearson']:.6f}")
    print(f"Val R²: {fold_results[best_fold_idx]['val_metrics']['r2']:.6f}")
    
    # Save best model
    artifacts_dir = project_dir / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    
    model_path = save_model(best_model, artifacts_dir / 'models' / 'ridge_grouped_model.pkl')
    scaler_path = save_scaler(best_scaler, artifacts_dir / 'scalers' / 'ridge_grouped_scaler.pkl')
    
    print(f"\n{'='*60}")
    print("SAVED ARTIFACTS")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    
    # Save results
    results = {
        'method': 'groupkfold_cv',
        'n_folds': n_folds,
        'config': config,
        'fold_results': fold_results,
        'summary': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                    for k, v in metrics_summary.items()},
        'best_fold': best_fold_idx + 1,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'note': 'GroupKFold used to prevent temporal label leakage'
    }
    
    results_path = project_dir / 'results' / 'ridge_grouped_results.json'
    results_path.parent.mkdir(exist_ok=True, parents=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults: {results_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train Ridge with GroupKFold CV')
    parser.add_argument('--config', type=str, default='configs/ridge.yaml',
                       help='Path to config file')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of folds for GroupKFold CV')
    
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / args.config
    data_path = project_dir / 'data' / 'train_fe_grouped.parquet'
    
    # Check if grouped data exists
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found")
        print("Please run: python scripts/add_time_groups.py")
        return
    
    # Load config
    config = load_config(config_path)
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Config file: {config_path}")
    print(f"Data file: {data_path}")
    print(f"Number of folds: {args.cv}")
    
    # Load data
    df = load_data(data_path)
    
    # Verify time_group_id exists
    if 'time_group_id' not in df.columns:
        print("❌ Error: time_group_id column not found")
        print("Please run: python scripts/add_time_groups.py")
        return
    
    print(f"✓ time_group_id column present with {df['time_group_id'].nunique()} groups")
    
    # Train with GroupKFold
    results = train_with_group_kfold(df, args.cv, config, project_dir)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\n⚠️  IMPORTANT: This model was trained with GroupKFold")
    print("   to prevent temporal label leakage in time-series data.")
    print("   Validation scores are more realistic and generalizable.")

if __name__ == '__main__':
    main()
