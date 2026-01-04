#!/usr/bin/env python
"""
Train LightGBM model using configuration file with support for Group K-Fold CV.

Usage:
    python scripts/train_lightgbm.py [--config configs/lightgbm.yaml]
"""

import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
import json
import time

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

def train_single_split(df, config, project_dir):
    """Train model using a single train/validation split."""
    print("\n" + "="*60)
    print("TRAINING MODE: SINGLE SPLIT")
    print("="*60)
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        df,
        split_method=config['split']['method'],
        test_size=config['split']['test_size'],
        random_state=config['split']['random_state']
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples")
    
    # Train model
    model, metrics = train_lightgbm(
        X_train, y_train, X_val, y_val,
        params=config['model'],
        num_boost_round=config['training']['num_boost_round'],
        early_stopping_rounds=config['training']['early_stopping_rounds']
    )
    
    # Print evaluation results
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    print(f"Train  - Pearson: {train_metrics['pearson']:.6f}, RMSE: {train_metrics['rmse']:.6f}, MAE: {train_metrics['mae']:.6f}")
    print(f"Valid  - Pearson: {val_metrics['pearson']:.6f}, RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}")
    
    # Save model
    model_dir = project_dir / config['output']['model_dir']
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / config['output']['model_name']
    
    print(f"\nSaving model to {model_path}...")
    model.save_model(str(model_path))
    
    return model, {'train': train_metrics, 'val': val_metrics}

def train_with_cv(df, config, project_dir):
    """Train model using K-Fold or Group K-Fold cross-validation."""
    cv_method = config['split']['cv_method']
    n_splits = config['split']['n_splits']
    
    print("\n" + "="*60)
    print(f"TRAINING MODE: {cv_method.upper()} CROSS-VALIDATION")
    print("="*60)
    print(f"Number of folds: {n_splits}")
    
    # Get CV splits
    if cv_method == 'group_kfold':
        group_col = config['split']['group_col']
        print(f"Group column: {group_col}")
        splits = split_data(
            df,
            split_method='group_kfold',
            n_splits=n_splits,
            group_col=group_col
        )
    elif cv_method == 'kfold':
        splits = split_data(
            df,
            split_method='kfold',
            n_splits=n_splits,
            random_state=config['split'].get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")
    
    # Store results for each fold
    cv_results = []
    models = []
    
    # Train on each fold
    for fold_idx, X_train, X_val, y_train, y_val in splits:
        print("\n" + "="*60)
        print(f"FOLD {fold_idx}/{n_splits}")
        print("="*60)
        
        # Train model
        model, metrics = train_lightgbm(
            X_train, y_train, X_val, y_val,
            params=config['model'],
            num_boost_round=config['training']['num_boost_round'],
            early_stopping_rounds=config['training']['early_stopping_rounds'],
            min_delta=config['training']['min_delta']
        )
        
        # Extract metrics
        train_metrics = metrics['train']
        val_metrics = metrics['val']
        print(f"Fold {fold_idx} - Train Pearson: {train_metrics['pearson']:.6f}, Val Pearson: {val_metrics['pearson']:.6f}")
        
        cv_results.append({
            'fold': fold_idx,
            'train_pearson': train_metrics['pearson'],
            'val_pearson': val_metrics['pearson'],
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse'],
            'best_iteration': model.best_iteration
        })
        
        models.append(model)
    
    # Print CV summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    val_pearsons = [r['val_pearson'] for r in cv_results]
    val_rmses = [r['val_rmse'] for r in cv_results]
    best_iterations = [r['best_iteration'] for r in cv_results]
    
    print(f"Validation Pearson: {np.mean(val_pearsons):.6f} ± {np.std(val_pearsons):.6f}")
    print(f"Validation RMSE:    {np.mean(val_rmses):.6f} ± {np.std(val_rmses):.6f}")
    print(f"Best iterations:    {np.median(best_iterations):.0f} (median), {np.mean(best_iterations):.1f} (mean)")
    print(f"\nFold-by-fold results:")
    for r in cv_results:
        print(f"  Fold {r['fold']}: Pearson={r['val_pearson']:.6f}, RMSE={r['val_rmse']:.6f}, Iter={r['best_iteration']}")
    
    # Save CV results
    results_dir = project_dir / 'results' / 'lightgbm_cv'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with open(results_dir / 'cv_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nCV results saved to {results_dir / 'cv_results.json'}")
    
    # Save all fold models
    model_dir = project_dir / config['output']['model_dir']
    model_dir.mkdir(exist_ok=True, parents=True)
    
    for idx, model in enumerate(models, 1):
        fold_model_path = model_dir / f"lightgbm_model_fold{idx}.txt"
        model.save_model(str(fold_model_path))
    print(f"All {len(models)} fold models saved to {model_dir}")
    
    # Train final model on full data using median best_iteration
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL ON FULL DATA")
    print("="*60)
    
    optimal_rounds = int(np.median(best_iterations))
    print(f"Using {optimal_rounds} rounds (median of CV best iterations)")
    
    # Prepare full data
    X_full = df.drop(columns=[config['data']['target_col']])
    y_full = df[config['data']['target_col']]
    
    # Drop group column if exists
    if cv_method == 'group_kfold':
        group_col = config['split']['group_col']
        if group_col in X_full.columns:
            X_full = X_full.drop(columns=[group_col])
    
    print(f"Full data: {X_full.shape[0]} samples, {X_full.shape[1]} features")
    
    # Create dataset and train without validation (no early stopping)
    train_data = lgb.Dataset(X_full, label=y_full)
    
    print(f"\nTraining final model for {optimal_rounds} iterations...")
    start_time = time.time()
    
    final_model = lgb.train(
        config['model'],
        train_data,
        num_boost_round=optimal_rounds,
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    training_time = time.time() - start_time
    print(f"\nFinal model training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = model_dir / config['output']['model_name']
    print(f"\nSaving final model to {final_model_path}...")
    final_model.save_model(str(final_model_path))
    
    return final_model, cv_results

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
    
    # Determine training mode
    use_cv = config['split'].get('use_cv', False)
    
    if use_cv:
        # Cross-validation mode
        models, cv_results = train_with_cv(df, config, project_dir)
    else:
        # Single split mode
        model, metrics = train_single_split(df, config, project_dir)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
