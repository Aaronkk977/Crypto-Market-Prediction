#!/usr/bin/env python
"""
LightGBM Feature Ablation Study based on Ridge Permutation Importance.

This script tests LightGBM performance with different numbers of top-ranked features
from Ridge's permutation importance analysis.

Usage:
    python scripts/lightgbm_feature_ablation.py                    # Test default feature counts
    python scripts/lightgbm_feature_ablation.py --top 30 50 100    # Test specific counts
    python scripts/lightgbm_feature_ablation.py --cv 3             # Use 3-fold CV (faster)
"""

import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
import pandas as pd
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import train_lightgbm

try:
    import lightgbm as lgb
except ImportError:
    print("Error: LightGBM is not installed.")
    print("Install with: pip install lightgbm")
    sys.exit(1)

def load_feature_importance(importance_path):
    """Load feature importance ranking from Ridge permutation importance."""
    df = pd.read_csv(importance_path)
    # Sort by importance_mean (descending)
    df = df.sort_values('importance_mean', ascending=False)
    return df['feature'].tolist()

def run_ablation_experiment(df, feature_list, top_n, config, n_folds=3):
    """
    Run LightGBM training with top N features from importance ranking.
    
    Args:
        df: Full dataframe
        feature_list: Ordered list of features by importance
        top_n: Number of top features to use (None = all features)
        config: Configuration dict
        n_folds: Number of CV folds
        
    Returns:
        dict: Results including metrics and feature count
    """
    print("\n" + "="*60)
    if top_n is None:
        print(f"TESTING WITH ALL FEATURES")
        selected_features = feature_list
    else:
        print(f"TESTING WITH TOP {top_n} FEATURES")
        selected_features = feature_list[:top_n]
    print("="*60)
    
    # Filter dataframe to selected features + target + group_col
    target_col = config['data']['target_col']
    group_col = config['split']['group_col']
    
    available_features = [f for f in selected_features if f in df.columns]
    print(f"Available features: {len(available_features)}/{len(selected_features)}")
    
    cols_to_keep = available_features + [target_col, group_col]
    df_filtered = df[cols_to_keep].copy()
    
    # Get CV splits
    cv_method = config['split']['cv_method']
    splits = split_data(
        df_filtered,
        split_method=cv_method,
        n_splits=n_folds,
        group_col=group_col,
        target_col=target_col
    )
    
    # Store results for each fold
    fold_results = []
    
    # Train on each fold
    for fold_idx, X_train, X_val, y_train, y_val in splits:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'='*60}")
        
        # Train model
        model, metrics = train_lightgbm(
            X_train, y_train, X_val, y_val,
            params=config['model'],
            num_boost_round=config['training']['num_boost_round'],
            early_stopping_rounds=config['training']['early_stopping_rounds']
        )
        
        fold_results.append({
            'fold': fold_idx,
            'train_pearson': metrics['train']['pearson'],
            'val_pearson': metrics['val']['pearson'],
            'train_rmse': metrics['train']['rmse'],
            'val_rmse': metrics['val']['rmse'],
            'best_iteration': model.best_iteration
        })
        
        print(f"Fold {fold_idx} - Val Pearson: {metrics['val']['pearson']:.6f}, Val RMSE: {metrics['val']['rmse']:.6f}")
    
    # Calculate average metrics
    val_pearsons = [r['val_pearson'] for r in fold_results]
    val_rmses = [r['val_rmse'] for r in fold_results]
    best_iterations = [r['best_iteration'] for r in fold_results]
    
    results = {
        'n_features': len(available_features),
        'top_n': top_n,
        'val_pearson_mean': float(np.mean(val_pearsons)),
        'val_pearson_std': float(np.std(val_pearsons)),
        'val_rmse_mean': float(np.mean(val_rmses)),
        'val_rmse_std': float(np.std(val_rmses)),
        'best_iteration_median': int(np.median(best_iterations)),
        'fold_results': fold_results
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR TOP {top_n if top_n else 'ALL'} FEATURES")
    print(f"{'='*60}")
    print(f"Number of features: {len(available_features)}")
    print(f"Val Pearson: {results['val_pearson_mean']:.6f} ± {results['val_pearson_std']:.6f}")
    print(f"Val RMSE:    {results['val_rmse_mean']:.6f} ± {results['val_rmse_std']:.6f}")
    print(f"Best iterations (median): {results['best_iteration_median']}")
    
    return results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='LightGBM Feature Ablation Study')
    parser.add_argument('--config', type=str, default='configs/lightgbm.yaml',
                        help='Path to LightGBM config file')
    parser.add_argument('--importance', type=str, 
                        default='results/importance/permutation_importance.csv',
                        help='Path to permutation importance CSV')
    parser.add_argument('--top', type=int, nargs='+', 
                        default=[30, 50, 100, 200, 500],
                        help='List of top N features to test (default: 30 50 100 200 500)')
    parser.add_argument('--cv', type=int, default=3,
                        help='Number of CV folds (default: 3)')
    parser.add_argument('--include-all', action='store_true',
                        help='Also test with all features')
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / args.config
    importance_path = project_dir / args.importance
    
    print("\n" + "="*60)
    print("LIGHTGBM FEATURE ABLATION STUDY")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Importance ranking: {importance_path}")
    print(f"CV folds: {args.cv}")
    print(f"Testing top N: {args.top}")
    if args.include_all:
        print("Including test with ALL features")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load feature importance ranking
    if not importance_path.exists():
        print(f"\n❌ Error: Importance file not found at {importance_path}")
        print("Please run: python scripts/permutation_importance.py")
        return
    
    feature_ranking = load_feature_importance(importance_path)
    print(f"\nLoaded feature ranking with {len(feature_ranking)} features")
    print(f"Top 5 features: {feature_ranking[:5]}")
    
    # Load data
    data_path = project_dir / config['data']['train_path']
    df = load_data(data_path)
    
    # Verify group column exists
    group_col = config['split']['group_col']
    if group_col not in df.columns:
        print(f"\n❌ Error: {group_col} column not found")
        print("Please run: python scripts/add_time_groups.py")
        return
    
    # Run ablation experiments
    all_results = []
    start_time = time.time()
    
    # Test with different numbers of top features
    for top_n in args.top:
        if top_n > len(feature_ranking):
            print(f"\n⚠️  Warning: top_n={top_n} exceeds available features ({len(feature_ranking)}), using all")
            top_n = None
        
        results = run_ablation_experiment(df, feature_ranking, top_n, config, args.cv)
        all_results.append(results)
    
    # Optionally test with all features
    if args.include_all and (max(args.top) < len(feature_ranking)):
        results = run_ablation_experiment(df, feature_ranking, None, config, args.cv)
        all_results.append(results)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"{'N Features':<12} {'Val Pearson':<20} {'Val RMSE':<20} {'Best Iter'}")
    print("-" * 60)
    
    for result in all_results:
        n_feat = result['n_features']
        pearson = f"{result['val_pearson_mean']:.6f} ± {result['val_pearson_std']:.6f}"
        rmse = f"{result['val_rmse_mean']:.6f} ± {result['val_rmse_std']:.6f}"
        best_iter = result['best_iteration_median']
        print(f"{n_feat:<12} {pearson:<20} {rmse:<20} {best_iter}")
    
    # Find best configuration
    best_idx = np.argmax([r['val_pearson_mean'] for r in all_results])
    best_result = all_results[best_idx]
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    print(f"Number of features: {best_result['n_features']}")
    print(f"Val Pearson: {best_result['val_pearson_mean']:.6f} ± {best_result['val_pearson_std']:.6f}")
    print(f"Val RMSE: {best_result['val_rmse_mean']:.6f} ± {best_result['val_rmse_std']:.6f}")
    
    # Save results
    results_dir = project_dir / 'results' / 'lightgbm_feature_ablation'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    output_data = {
        'experiment_config': {
            'cv_folds': args.cv,
            'tested_top_n': args.top,
            'total_features': len(feature_ranking),
            'config_file': str(config_path)
        },
        'results': all_results,
        'best_config': {
            'n_features': best_result['n_features'],
            'top_n': best_result['top_n'],
            'val_pearson': best_result['val_pearson_mean'],
            'val_pearson_std': best_result['val_pearson_std']
        },
        'total_time_seconds': total_time
    }
    
    # Save JSON
    json_path = results_dir / 'ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Save comparison CSV
    comparison_df = pd.DataFrame([
        {
            'n_features': r['n_features'],
            'top_n': r['top_n'],
            'val_pearson_mean': r['val_pearson_mean'],
            'val_pearson_std': r['val_pearson_std'],
            'val_rmse_mean': r['val_rmse_mean'],
            'val_rmse_std': r['val_rmse_std'],
            'best_iteration_median': r['best_iteration_median']
        }
        for r in all_results
    ])
    
    csv_path = results_dir / 'ablation_comparison.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to {csv_path}")
    
    print(f"\nTotal experiment time: {total_time:.1f} seconds")
    print("\n" + "="*60)
    print("✅ ABLATION STUDY COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
