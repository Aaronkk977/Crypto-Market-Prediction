#!/usr/bin/env python
"""
LightGBM Hyperparameter Tuning Experiment.

Usage:
    python scripts/lightgbm_tuning.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import train_lightgbm
from myproj.metrics import calculate_metrics
import lightgbm as lgb


def save_results(results_list, output_dir):
    """Save experiment results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save JSON
    with open(output_dir / 'lightgbm_tuning_results.json', 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Create comparison table
    comparison = pd.DataFrame([{
        'Experiment': r['experiment'],
        'Learning Rate': r['learning_rate'],
        'Num Leaves': r['num_leaves'],
        'Best Iteration': r['best_iteration'],
        'Train R²': r['train_metrics']['r2'],
        'Train Pearson': r['train_metrics']['pearson'],
        'Val R²': r['val_metrics']['r2'],
        'Val Pearson': r['val_metrics']['pearson'],
        'Val RMSE': r['val_metrics']['rmse'],
        'Time (s)': r['training_time']
    } for r in results_list])
    
    comparison.to_csv(output_dir / 'lightgbm_tuning_comparison.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    return comparison

def main():
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'lightgbm'
    model_dir = project_dir / 'artifacts' / 'models'
    
    print("\n" + "="*60)
    print("LIGHTGBM HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    train_path = data_dir / 'train_fe.parquet'
    df = load_data(train_path)
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        df,
        split_method='random',
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples")
    
    # Hyperparameter configurations to test
    configs = [
        {'name': 'Fast', 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1},
        {'name': 'Balanced', 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1},
        {'name': 'Deep', 'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 10},
        {'name': 'Conservative', 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 5},
        {'name': 'Aggressive', 'learning_rate': 0.1, 'num_leaves': 127, 'max_depth': 15},
    ]
    
    # Base parameters
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1
    }
    
    # Run experiments
    results = []
    best_val_pearson = -np.inf
    best_model = None
    best_config_name = None
    
    import time
    
    for config in configs:
        experiment_name = f"LightGBM ({config['name']})"
        print(f"\n{'='*60}")
        print(f"{experiment_name}")
        print(f"{'='*60}")
        
        # Merge config with base params
        params = base_params.copy()
        params.update({
            'learning_rate': config['learning_rate'],
            'num_leaves': config['num_leaves'],
            'max_depth': config['max_depth']
        })
        
        start_time = time.time()
        
        # Train model
        model = train_lightgbm(
            X_train, y_train, X_val, y_val,
            params=params,
            num_boost_round=1000,
            early_stopping_rounds=50
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        
        result = {
            'experiment': experiment_name,
            'learning_rate': config['learning_rate'],
            'num_leaves': config['num_leaves'],
            'max_depth': config['max_depth'],
            'best_iteration': int(model.best_iteration),
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        results.append(result)
        
        # Track best model
        val_pearson = val_metrics['pearson']
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_model = model
            best_config_name = config['name']
    
    # Save results
    comparison = save_results(results, results_dir)
    
    # Save best model
    print(f"\n{'='*60}")
    print("Saving best model...")
    print(f"{'='*60}")
    print(f"Best configuration: {best_config_name}")
    print(f"Best validation Pearson: {best_val_pearson:.6f}")
    
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / 'lightgbm_best_model.txt'
    best_model.save_model(str(model_path))
    print(f"Model saved to {model_path}")
    
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
    
    print(f"\nBest Val R²: {comparison.loc[best_r2_idx, 'Experiment']} "
          f"(R²={comparison.loc[best_r2_idx, 'Val R²']:.6f})")
    print(f"Best Val Pearson: {comparison.loc[best_pearson_idx, 'Experiment']} "
          f"(Pearson={comparison.loc[best_pearson_idx, 'Val Pearson']:.6f})")
    print(f"Best Val RMSE: {comparison.loc[best_rmse_idx, 'Experiment']} "
          f"(RMSE={comparison.loc[best_rmse_idx, 'Val RMSE']:.6f})")
    
    print("\n" + "="*60)
    print("LightGBM tuning experiment completed!")
    print("="*60)

if __name__ == "__main__":
    main()
