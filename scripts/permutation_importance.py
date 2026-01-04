#!/usr/bin/env python
"""
Calculate permutation importance for trained model (Ridge or LightGBM).

Usage:
    python scripts/permutation_importance.py --model lightgbm
    python scripts/permutation_importance.py --model ridge --data train_top200_grouped.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
import argparse
import lightgbm as lgb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data, get_feature_groups
from myproj.split import split_data
from myproj.utils import load_model, load_scaler
from myproj.metrics import get_pearson_scorer, pearson_correlation
import time


class LGBMWrapper:
    """Wrapper to make LightGBM Booster compatible with sklearn's permutation_importance."""
    def __init__(self, booster):
        self.booster = booster
    
    def predict(self, X):
        return self.booster.predict(X)
    
    def fit(self, X, y):
        # Not used, but required by sklearn interface
        pass

def main():
    parser = argparse.ArgumentParser(description='Calculate permutation importance')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'ridge'],
                        help='Model type (default: lightgbm)')
    parser.add_argument('--data', type=str, default='train_top200_grouped.parquet',
                        help='Input data file (default: train_top200_grouped.parquet)')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Model file name (default: lightgbm_model.txt for lightgbm, ridge_model.pkl for ridge)')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of permutation repeats (default: 10)')
    parser.add_argument('--use_val_fold', type=int, default=0,
                        help='Which fold to use for validation (0-4, default: 0)')
    
    args = parser.parse_args()
    
    # Set up paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    model_dir = project_dir / 'artifacts' / 'models'
    scaler_dir = project_dir / 'artifacts' / 'scalers'
    output_dir = project_dir / 'results' / f'{args.model}_importance'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print(f"PERMUTATION IMPORTANCE ANALYSIS - {args.model.upper()}")
    print("="*60)
    
    # Load data
    train_path = data_dir / args.data
    if not train_path.exists():
        print(f"\n❌ Error: {train_path.name} not found!")
        return
    
    print(f"\n📂 Loading data from {train_path.name}...")
    df = load_data(train_path)
    
    # Get feature columns and label
    metadata_cols = ['symbol', 'date_id', 'time_id', 'time_group_id']
    label_cols = ['label', 'Y']
    feature_cols = [col for col in df.columns if col not in metadata_cols + label_cols]
    label_col = 'label' if 'label' in df.columns else 'Y'
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Total samples: {len(df)}")
    
    # Prepare data for GroupKFold validation
    X = df[feature_cols].values
    y = df[label_col].values
    
    if 'time_group_id' in df.columns:
        groups = df['time_group_id'].values
        print(f"  Using time_group_id for validation ({len(np.unique(groups))} groups)")
    else:
        print("  Warning: time_group_id not found, using sequential grouping")
        groups = np.arange(len(df)) // 100
    
    # Split using GroupKFold
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = list(gkf.split(X, y, groups))[args.use_val_fold]
    
    X_val = X[val_idx]
    y_val = y[val_idx]
    feature_names = feature_cols
    
    print(f"  Validation fold {args.use_val_fold}: {len(val_idx)} samples")
    
    # Load model
    if args.model == 'lightgbm':
        model_file = args.model_file or 'lightgbm_model.txt'
        model_path = model_dir / model_file
        if not model_path.exists():
            print(f"\n❌ Error: Model file not found: {model_path}")
            return
        print(f"\n📦 Loading LightGBM model from {model_file}...")
        booster = lgb.Booster(model_file=str(model_path))
        model = LGBMWrapper(booster)  # Wrap for sklearn compatibility
        scaler = None
    else:  # ridge
        model_file = args.model_file or 'ridge_model.pkl'
        model = load_model(model_dir / model_file)
        scaler = load_scaler(scaler_dir / 'scaler.pkl')
        print(f"\n📦 Loading Ridge model from {model_file}...")
        print("Scaling features...")
        X_val = scaler.transform(X_val)
    
    # Calculate permutation importance
    print("\nCalculating permutation importance...")
    print(f"(This may take a few minutes with {args.n_repeats} repeats...)")
    start_time = time.time()
    
    # Use Pearson correlation scorer
    pearson_score = get_pearson_scorer()
    
    perm_importance = permutation_importance(
        model, X_val, y_val,
        n_repeats=args.n_repeats, 
        random_state=42,
        scoring=pearson_score,
        n_jobs=4
    )
    
    calc_time = time.time() - start_time
    print(f"Completed in {calc_time:.2f}s")
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Try to get feature groups (may not work for all datasets)
    try:
        _, engineered_features, _ = get_feature_groups(df)
        importance_df['is_engineered'] = importance_df['feature'].isin(engineered_features)
    except:
        importance_df['is_engineered'] = False
        engineered_features = []
    
    # Save results
    output_path = output_dir / f'permutation_importance_fold{args.use_val_fold}.csv'
    importance_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Top 20 Most Important Features:")
    print("="*60)
    for idx, row in importance_df.head(20).iterrows():
        marker = "✓" if row['is_engineered'] else " "
        print(f"[{marker}] {row['feature']:30s}  {row['importance_mean']:8.6f} ± {row['importance_std']:.6f}")
    
    # Engineered features summary
    if len(engineered_features) > 0:
        print("\n" + "="*60)
        print("Engineered Features Summary:")
        print("="*60)
        eng_features = importance_df[importance_df['is_engineered']]
        
        print(f"Total engineered features: {len(eng_features)}")
        print(f"Average importance: {eng_features['importance_mean'].mean():.6f}")
        print(f"Near-zero importance (<0.0001): {(eng_features['importance_mean'].abs() < 0.0001).sum()}")
        
        print("\nEngineered features by importance:")
        for idx, row in eng_features.iterrows():
            print(f"  {row['feature']:30s}  {row['importance_mean']:8.6f} ± {row['importance_std']:.6f}")
    
    print("\n" + "="*60)
    print("Permutation importance analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main()
