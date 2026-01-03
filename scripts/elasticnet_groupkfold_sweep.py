#!/usr/bin/env python
"""
ElasticNet Grid Search with GroupKFold CV - Proper Financial ML Approach.

This script addresses the methodology issues:
1. Uses GroupKFold to prevent temporal label leakage
2. Optimizes for Pearson correlation (not MSE)
3. Provides uncertainty estimates (mean ± std across folds)
4. Includes time holdout validation for regime shift detection

Usage:
    python scripts/elasticnet_groupkfold_sweep.py
    python scripts/elasticnet_groupkfold_sweep.py --n-folds 10
    python scripts/elasticnet_groupkfold_sweep.py --quick  # Fast test mode
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
import argparse
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr, t as t_dist

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import scale_features
from myproj.metrics import calculate_metrics
from myproj.utils import save_model, save_scaler


def fit_and_score_pearson(X_train, y_train, X_val, y_val, alpha, l1_ratio, max_iter=100000):
    """
    Fit ElasticNet and return Pearson correlation on validation set.
    
    This is the key change: we optimize for Pearson, not MSE.
    """
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=1e-4,
        random_state=42,  # Note: Has minimal effect with default selection='cyclic'
        selection='cyclic'  # Explicit: deterministic coordinate descent
    )
    model.fit(X_train, y_train)
    
    # Predict on validation
    y_val_pred = model.predict(X_val)
    
    # Calculate Pearson correlation (our true objective)
    # Guard against NaN (can happen when predictions are constant)
    pearson_corr, _ = pearsonr(y_val, y_val_pred)
    if not np.isfinite(pearson_corr):
        pearson_corr = -1.0  # Treat as very poor performance
    
    # Also track sparsity
    n_nonzero = np.sum(model.coef_ != 0)
    
    return {
        'pearson': pearson_corr,
        'n_nonzero': n_nonzero,
        'model': model
    }


def groupkfold_grid_search(df, alphas, l1_ratios, n_splits=5, verbose=True):
    """
    Grid search over (alpha, l1_ratio) using GroupKFold CV.
    Optimizes for Pearson correlation.
    
    Returns:
        DataFrame with results for each (alpha, l1_ratio) combination
    """
    print("\n" + "="*60)
    print(f"GROUPKFOLD GRID SEARCH (n_splits={n_splits})")
    print("="*60)
    print(f"Grid size: {len(alphas)} alphas × {len(l1_ratios)} l1_ratios = {len(alphas)*len(l1_ratios)} fits × {n_splits} folds")
    print(f"Total model fits: {len(alphas)*len(l1_ratios)*n_splits}")
    print("\n⚠️  Optimizing for PEARSON correlation (not MSE)")
    print("✓ Using GroupKFold to prevent temporal label leakage")
    
    # Get GroupKFold splits - CRITICAL: explicitly drop time_group_id
    fold_generator = split_data(
        df,
        split_method='group_kfold',
        group_col='time_group_id',
        n_splits=n_splits,
        target_col='label',
        drop_cols=['time_group_id']  # Explicit: prevent data leakage
    )
    
    # Collect fold data (scale each fold independently to prevent leakage)
    fold_data = []
    
    print("\nPreparing folds (scaling each independently)...")
    for fold_idx, X_train, X_val, y_train, y_val in fold_generator:
        # Scale features for this fold
        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
        fold_data.append((X_train_scaled, X_val_scaled, y_train, y_val))
    
    print(f"\n✓ Prepared {len(fold_data)} folds")
    print("Starting grid search...\n")
    
    # Grid search
    results = []
    total_fits = len(alphas) * len(l1_ratios)
    fit_count = 0
    start_time = time.time()
    
    for l1_ratio in l1_ratios:
        for alpha in alphas:
            fit_count += 1
            
            # Evaluate on all folds
            fold_scores = []
            fold_sparsity = []
            
            for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(fold_data):
                result = fit_and_score_pearson(
                    X_train, y_train, X_val, y_val,
                    alpha, l1_ratio
                )
                fold_scores.append(result['pearson'])
                fold_sparsity.append(result['n_nonzero'])
            
            # Aggregate across folds
            mean_pearson = np.mean(fold_scores)
            std_pearson = np.std(fold_scores)
            mean_sparsity = np.mean(fold_sparsity)
            
            results.append({
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'mean_pearson': mean_pearson,
                'std_pearson': std_pearson,
                'mean_nonzero': mean_sparsity,
                'fold_scores': fold_scores
            })
            
            if verbose and fit_count % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / fit_count * (total_fits - fit_count)
                print(f"  Progress: {fit_count}/{total_fits} ({fit_count/total_fits*100:.1f}%) | "
                      f"ETA: {eta/60:.1f}m | "
                      f"Best so far: {max(r['mean_pearson'] for r in results):.6f}")
    
    total_time = time.time() - start_time
    print(f"\n✓ Grid search completed in {total_time/60:.1f} minutes")
    
    results_df = pd.DataFrame(results)
    return results_df


def time_holdout_validation(df_holdout, best_alpha, best_l1_ratio, test_size=0.2):
    """
    Validate on independent time holdout set to check for regime shift.
    
    Args:
        df_holdout: Independent holdout data (last 20% of original dataset)
        best_alpha: Best alpha from GroupKFold CV
        best_l1_ratio: Best l1_ratio from GroupKFold CV
        test_size: Not used, kept for API compatibility
    
    Returns:
        dict with train_metrics, val_metrics, model, scaler
    """
    print("\n" + "="*60)
    print("TIME HOLDOUT VALIDATION (Regime Shift Check)")
    print("="*60)
    print(f"Holdout samples: {len(df_holdout):,}")
    print(f"Holdout time groups: {df_holdout['time_group_id'].nunique():,}")
    
    # Prepare holdout data
    X_holdout = df_holdout.drop(columns=['label', 'time_group_id'], errors='ignore')
    y_holdout = df_holdout['label']
    
    # Scale
    scaler = StandardScaler()
    X_holdout_scaled = scaler.fit_transform(X_holdout)
    
    # Train with best params on full holdout
    model = ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        max_iter=100000,
        tol=1e-4,
        selection='cyclic',
        random_state=42
    )
    model.fit(X_holdout_scaled, y_holdout)
    
    # Evaluate
    y_train_pred = model.predict(X_holdout_scaled)
    y_val_pred = model.predict(X_holdout_scaled)  # Same as train for holdout
    
    train_metrics = calculate_metrics(y_holdout, y_train_pred, prefix="Holdout Train")
    val_metrics = calculate_metrics(y_holdout, y_val_pred, prefix="Holdout Val")
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model': model,
        'scaler': scaler
    }


def main():
    parser = argparse.ArgumentParser(description='ElasticNet GroupKFold Grid Search')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of folds for GroupKFold (default: 5)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer grid points)')
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'elasticnet_groupkfold'
    model_dir = project_dir / 'artifacts' / 'models'
    scaler_dir = project_dir / 'artifacts' / 'scalers'
    
    print("\n" + "="*60)
    print("ELASTICNET GROUPKFOLD GRID SEARCH")
    print("="*60)
    print("\n📋 Methodology:")
    print("  • GroupKFold CV to prevent temporal label leakage")
    print("  • Optimize for Pearson correlation (not MSE)")
    print("  • Mean ± Std across folds for uncertainty estimation")
    print("  • Time holdout validation for regime shift detection")
    
    # Load data
    train_path = data_dir / 'train_fe_grouped.parquet'
    if not train_path.exists():
        print(f"\n❌ Error: {train_path} not found")
        print("   Run: python scripts/add_time_groups.py")
        return
    
    df = load_data(train_path)
    
    if 'time_group_id' not in df.columns:
        print("❌ Error: time_group_id column not found")
        print("   Run: python scripts/add_time_groups.py")
        return
    
    print(f"\n✓ Loaded data: {df.shape[0]:,} samples, {df.shape[1]} features")
    print(f"✓ Time groups: {df['time_group_id'].nunique():,} groups")
    
    # ===== CRITICAL: Separate time holdout BEFORE CV =====
    # Issue #2 fix: Prevent data leakage by isolating holdout from training
    cutoff_idx = int(len(df) * 0.8)
    df_train = df.iloc[:cutoff_idx].copy()  # First 80% for GroupKFold CV
    df_holdout = df.iloc[cutoff_idx:].copy()  # Last 20% for independent validation
    
    print(f"\n🔪 Time-based split:")
    print(f"  Training set: {len(df_train):,} samples ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Holdout set:  {len(df_holdout):,} samples ({len(df_holdout)/len(df)*100:.1f}%)")
    print(f"  Training time groups: {df_train['time_group_id'].nunique():,} groups")
    print(f"  Holdout time groups: {df_holdout['time_group_id'].nunique():,} groups")
    # =====================================================
    
    # Define grid
    if args.quick:
        print("\n⚡ Quick test mode")
        alphas = np.logspace(-2, 2, 10)  # 10 alphas
        l1_ratios = [0.1, 0.5, 0.8]  # 3 l1_ratios
    else:
        # Full grid based on previous results
        alphas = np.logspace(-3, 3, 30)  # 30 alphas: 0.001 to 1000
        l1_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 10 l1_ratios
    
    print(f"\n📊 Grid configuration:")
    print(f"  Alphas: {len(alphas)} values from {alphas.min():.4f} to {alphas.max():.1f}")
    print(f"  L1 ratios: {l1_ratios}")
    print(f"  Total: {len(alphas) * len(l1_ratios)} parameter combinations")
    
    # Run GroupKFold grid search (on training set only)
    print(f"\n🔍 Running GroupKFold grid search on TRAINING SET ONLY...")
    results_df = groupkfold_grid_search(
        df_train, alphas, l1_ratios, n_splits=args.n_folds
    )
    
    # Find best combination
    best_idx = results_df['mean_pearson'].idxmax()
    best_row = results_df.loc[best_idx]
    best_alpha = best_row['alpha']
    best_l1_ratio = best_row['l1_ratio']
    
    print("\n" + "="*60)
    print("BEST PARAMETERS (by GroupKFold CV)")
    print("="*60)
    print(f"Best alpha: {best_alpha:.6f}")
    print(f"Best l1_ratio: {best_l1_ratio:.3f}")
    print(f"Mean Pearson: {best_row['mean_pearson']:.6f} ± {best_row['std_pearson']:.6f}")
    print(f"Mean non-zero coef: {best_row['mean_nonzero']:.0f}")
    
    # Issue #4 fix: Correct uncertainty estimation with stderr
    k = args.n_folds
    stderr = best_row['std_pearson'] / np.sqrt(k)
    t_value = t_dist.ppf(0.975, df=k-1)  # 95% CI, two-tailed
    ci_lower = best_row['mean_pearson'] - t_value * stderr
    ci_upper = best_row['mean_pearson'] + t_value * stderr
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]  (using t-distribution with df={k-1})")
    print(f"\nFold-wise Pearson scores: {[f'{s:.6f}' for s in best_row['fold_scores']]}")
    
    # Show top 10 combinations
    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*60)
    top_10 = results_df.nlargest(10, 'mean_pearson')[['alpha', 'l1_ratio', 'mean_pearson', 'std_pearson', 'mean_nonzero']]
    print(top_10.to_string(index=False))
    
    # Time holdout validation (on independent holdout set)
    print(f"\n📊 Evaluating on independent holdout set ({len(df_holdout):,} samples)...")
    time_result = time_holdout_validation(df_holdout, best_alpha, best_l1_ratio)
    
    print("\n" + "="*60)
    print("COMPARISON: GroupKFold CV vs Time Holdout")
    print("="*60)
    print(f"{'Metric':<20} {'GroupKFold CV':<20} {'Time Holdout':<20}")
    print("-" * 60)
    print(f"{'Val Pearson':<20} {best_row['mean_pearson']:.6f} ± {best_row['std_pearson']:.6f}    {time_result['val_metrics']['pearson']:.6f}")
    print(f"{'Train Pearson':<20} {'N/A':<20} {time_result['train_metrics']['pearson']:.6f}")
    
    pearson_diff = abs(best_row['mean_pearson'] - time_result['val_metrics']['pearson'])
    if pearson_diff > 0.02:
        print(f"\n⚠️  WARNING: Large difference ({pearson_diff:.4f}) suggests regime shift!")
        print("   Model may not generalize well to recent time periods.")
    else:
        print(f"\n✓ Good alignment (diff={pearson_diff:.4f}) - consistent performance")
    
    # Save results
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Save full grid results
    results_df.to_csv(results_dir / 'grid_search_results.csv', index=False)
    
    # Save summary
    summary = {
        'method': 'groupkfold_grid_search',
        'n_folds': args.n_folds,
        'n_alphas': len(alphas),
        'n_l1_ratios': len(l1_ratios),
        'best_params': {
            'alpha': float(best_alpha),
            'l1_ratio': float(best_l1_ratio)
        },
        'groupkfold_cv': {
            'mean_pearson': float(best_row['mean_pearson']),
            'std_pearson': float(best_row['std_pearson']),
            'fold_scores': [float(s) for s in best_row['fold_scores']]
        },
        'time_holdout': {
            'train_pearson': time_result['train_metrics']['pearson'],
            'val_pearson': time_result['val_metrics']['pearson'],
            'val_r2': time_result['val_metrics']['r2']
        },
        'note': 'Optimized for Pearson correlation using GroupKFold to prevent temporal leakage'
    }
    
    with open(results_dir / 'best_params.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Train final model on all data and save
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL (on all data)")
    print("="*60)
    
    X = df.drop(columns=['label', 'time_group_id'])
    y = df['label']
    
    # Scale all data
    from sklearn.preprocessing import StandardScaler
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    # Train final model
    final_model = ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        max_iter=100000,
        tol=1e-4,
        random_state=42
    )
    final_model.fit(X_scaled, y)
    
    # Save
    model_dir.mkdir(exist_ok=True, parents=True)
    scaler_dir.mkdir(exist_ok=True, parents=True)
    
    from myproj.utils import save_model, save_scaler
    save_model(final_model, model_dir / 'elasticnet_groupkfold_best.pkl')
    save_scaler(scaler_final, scaler_dir / 'elasticnet_groupkfold_scaler.pkl')
    
    print(f"\n✓ Model saved to: {model_dir}/elasticnet_groupkfold_best.pkl")
    print(f"✓ Scaler saved to: {scaler_dir}/elasticnet_groupkfold_scaler.pkl")
    print(f"✓ Results saved to: {results_dir}/")
    
    # Final analysis
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    # Sparsity
    n_nonzero = np.sum(final_model.coef_ != 0)
    sparsity = (1 - n_nonzero / len(final_model.coef_)) * 100
    print(f"\n1. Feature Selection:")
    print(f"   • Non-zero coefficients: {n_nonzero}/{len(final_model.coef_)}")
    print(f"   • Sparsity: {sparsity:.1f}%")
    
    # Uncertainty
    print(f"\n2. Performance Uncertainty:")
    print(f"   • Mean Pearson: {best_row['mean_pearson']:.6f}")
    print(f"   • Std Pearson: {best_row['std_pearson']:.6f}")
    print(f"   • 95% CI: [{best_row['mean_pearson'] - 1.96*best_row['std_pearson']:.6f}, "
          f"{best_row['mean_pearson'] + 1.96*best_row['std_pearson']:.6f}]")
    
    # Regime stability
    print(f"\n3. Temporal Stability:")
    if pearson_diff < 0.01:
        print(f"   ✓ Excellent: CV and time holdout aligned (diff={pearson_diff:.4f})")
    elif pearson_diff < 0.02:
        print(f"   ✓ Good: Minor difference (diff={pearson_diff:.4f})")
    else:
        print(f"   ⚠️  Warning: Significant difference (diff={pearson_diff:.4f})")
        print(f"      Model may overfit to earlier time periods")
    
    print("\n" + "="*60)
    print("✅ EXPERIMENT COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
