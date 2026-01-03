#!/usr/bin/env python
"""
Train Elastic Net model with L1 ratio sweep.

Usage:
    python scripts/train_elasticnet.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import train_elasticnet_cv, scale_features
from myproj.metrics import calculate_metrics
from myproj.utils import save_model, save_scaler

def run_elasticnet_experiment(X_train, X_val, y_train, y_val, l1_ratio, experiment_name):
    """
    Run single ElasticNet experiment with specified L1 ratio.
    
    Args:
        X_train, X_val, y_train, y_val: Split datasets
        l1_ratio: L1 ratio (0=Ridge, 1=Lasso)
        experiment_name: Name of the experiment
    
    Returns:
        Dictionary with metrics and model
    """
    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print(f"{'='*60}")
    
    # Scale features
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    
    # Train model
    start_time = time.time()
    model = train_elasticnet_cv(X_train_scaled, y_train, l1_ratio=l1_ratio, cv=3, n_jobs=4)
    training_time = time.time() - start_time
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    train_metrics = calculate_metrics(y_train, y_train_pred, prefix="Train")
    val_metrics = calculate_metrics(y_val, y_val_pred, prefix="Validation")
    
    results = {
        'experiment': experiment_name,
        'l1_ratio': l1_ratio,
        'n_features': X_train.shape[1],
        'best_alpha': float(model.alpha_),
        'n_nonzero_coef': int(np.sum(model.coef_ != 0)),
        'training_time': training_time,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    
    return results, model, scaler

def save_results(results_list, output_dir):
    """Save experiment results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save JSON
    with open(output_dir / 'elasticnet_results.json', 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Create comparison table
    comparison = pd.DataFrame([{
        'L1 Ratio': r['l1_ratio'],
        'Best Alpha': r['best_alpha'],
        'Non-zero Coef': r['n_nonzero_coef'],
        'Train R²': r['train_metrics']['r2'],
        'Train Pearson': r['train_metrics']['pearson'],
        'Val R²': r['val_metrics']['r2'],
        'Val Pearson': r['val_metrics']['pearson'],
        'Val RMSE': r['val_metrics']['rmse'],
        'Val MAE': r['val_metrics']['mae'],
        'Time (s)': r['training_time']
    } for r in results_list])
    
    comparison.to_csv(output_dir / 'elasticnet_comparison.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    return comparison

def main():
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'elasticnet'
    model_dir = project_dir / 'artifacts' / 'models'
    scaler_dir = project_dir / 'artifacts' / 'scalers'
    
    print("\n" + "="*60)
    print("ELASTIC NET EXPERIMENT: L1 RATIO SWEEP")
    print("="*60)
    
    # Load data
    train_path = data_dir / 'train_fe.parquet'
    df = load_data(train_path)
    
    # Split data (use same split as other experiments)
    X_train, X_val, y_train, y_val = split_data(
        df,
        split_method='random',
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples")
    
    # L1 ratios to sweep
    l1_ratios = [0.05, 0.1, 0.2, 0.5, 0.8]
    
    print(f"\nL1 ratios to test: {l1_ratios}")
    print("Note: L1 ratio=0 is Ridge, L1 ratio=1 is Lasso")
    
    # Run experiments
    results = []
    best_val_pearson = -np.inf
    best_model = None
    best_scaler = None
    best_l1_ratio = None
    
    for l1_ratio in l1_ratios:
        experiment_name = f"Elastic Net (L1 ratio={l1_ratio})"
        
        result, model, scaler = run_elasticnet_experiment(
            X_train, X_val, y_train, y_val,
            l1_ratio, experiment_name
        )
        results.append(result)
        
        # Track best model
        val_pearson = result['val_metrics']['pearson']
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_model = model
            best_scaler = scaler
            best_l1_ratio = l1_ratio
    
    # Save results
    comparison = save_results(results, results_dir)
    
    # Save best model
    print(f"\n{'='*60}")
    print("Saving best model...")
    print(f"{'='*60}")
    print(f"Best L1 ratio: {best_l1_ratio}")
    print(f"Best validation Pearson: {best_val_pearson:.6f}")
    
    model_dir.mkdir(exist_ok=True, parents=True)
    scaler_dir.mkdir(exist_ok=True, parents=True)
    
    save_model(best_model, model_dir / 'elasticnet_model.pkl')
    save_scaler(best_scaler, scaler_dir / 'elasticnet_scaler.pkl')
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(comparison.to_string(index=False))
    
    # Find best by different metrics
    print("\n" + "="*60)
    print("BEST MODELS BY METRIC")
    print("="*60)
    
    best_r2_idx = comparison['Val R²'].idxmax()
    best_pearson_idx = comparison['Val Pearson'].idxmax()
    best_rmse_idx = comparison['Val RMSE'].idxmin()
    
    print(f"\nBest Val R²: L1 ratio={comparison.loc[best_r2_idx, 'L1 Ratio']:.2f} "
          f"(R²={comparison.loc[best_r2_idx, 'Val R²']:.6f})")
    print(f"Best Val Pearson: L1 ratio={comparison.loc[best_pearson_idx, 'L1 Ratio']:.2f} "
          f"(Pearson={comparison.loc[best_pearson_idx, 'Val Pearson']:.6f})")
    print(f"Best Val RMSE: L1 ratio={comparison.loc[best_rmse_idx, 'L1 Ratio']:.2f} "
          f"(RMSE={comparison.loc[best_rmse_idx, 'Val RMSE']:.6f})")
    
    # Sparsity analysis
    print("\n" + "="*60)
    print("SPARSITY ANALYSIS")
    print("="*60)
    for _, row in comparison.iterrows():
        sparsity = (1 - row['Non-zero Coef'] / X_train.shape[1]) * 100
        print(f"L1 ratio={row['L1 Ratio']:.2f}: {row['Non-zero Coef']}/{X_train.shape[1]} "
              f"non-zero ({sparsity:.1f}% sparse)")
    
    print("\n" + "="*60)
    print("Elastic Net experiment completed!")
    print("="*60)

if __name__ == "__main__":
    main()
