#!/usr/bin/env python
"""
Feature Correlation Filtering for Crypto Market Prediction.

This script:
1. Calculates Pearson correlation between all X1-X780 features
2. Identifies feature pairs with correlation > 0.995
3. Provides options to:
   - Remove redundant features (keep one from each pair)
   - Create residual signals (regress one on the other, keep residuals)
4. Saves filtered dataset and correlation analysis results

Usage:
    python scripts/filter_correlated_features.py
    python scripts/filter_correlated_features.py --threshold 0.995 --method remove
    python scripts/filter_correlated_features.py --method residual
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data


def calculate_correlation_matrix(df, feature_cols):
    """
    Calculate Pearson correlation matrix for specified features.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
    
    Returns:
        correlation_matrix: DataFrame with pairwise correlations
    """
    print(f"\n📊 Calculating correlation matrix for {len(feature_cols)} features...")
    
    # Use pandas corr for efficiency
    corr_matrix = df[feature_cols].corr(method='pearson')
    
    print(f"✓ Correlation matrix computed: {corr_matrix.shape}")
    return corr_matrix


def find_high_correlation_pairs(corr_matrix, threshold=0.995):
    """
    Find feature pairs with correlation above threshold.
    
    Args:
        corr_matrix: Correlation matrix (DataFrame)
        threshold: Correlation threshold (default: 0.995)
    
    Returns:
        high_corr_pairs: List of (feature1, feature2, correlation)
    """
    print(f"\n🔍 Finding feature pairs with |correlation| > {threshold}...")
    
    high_corr_pairs = []
    
    # Iterate through upper triangle only (avoid duplicates)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) > threshold:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_val))
    
    # Sort by absolute correlation (descending)
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"✓ Found {len(high_corr_pairs)} feature pairs with |correlation| > {threshold}")
    
    return high_corr_pairs


def select_features_to_remove(high_corr_pairs, feature_cols):
    """
    Select which features to remove from correlated pairs.
    
    Strategy: For each pair, keep the feature that appears in fewer pairs
    (i.e., remove the more redundant feature).
    
    Args:
        high_corr_pairs: List of (feature1, feature2, correlation)
        feature_cols: All feature column names
    
    Returns:
        features_to_remove: Set of feature names to remove
        features_to_keep: Set of feature names to keep
    """
    print("\n🎯 Selecting features to remove...")
    
    # Count how many pairs each feature appears in
    feature_pair_count = {}
    for feat1, feat2, _ in high_corr_pairs:
        feature_pair_count[feat1] = feature_pair_count.get(feat1, 0) + 1
        feature_pair_count[feat2] = feature_pair_count.get(feat2, 0) + 1
    
    features_to_remove = set()
    features_to_keep = set()
    
    for feat1, feat2, corr_val in high_corr_pairs:
        # Skip if both already processed
        if feat1 in features_to_remove or feat2 in features_to_remove:
            continue
        if feat1 in features_to_keep and feat2 in features_to_keep:
            continue
        
        # Remove the feature that appears in more pairs
        count1 = feature_pair_count.get(feat1, 0)
        count2 = feature_pair_count.get(feat2, 0)
        
        if count1 > count2:
            features_to_remove.add(feat1)
            features_to_keep.add(feat2)
        elif count2 > count1:
            features_to_remove.add(feat2)
            features_to_keep.add(feat1)
        else:
            # Equal counts: remove the one with higher index (arbitrary)
            if feat1 > feat2:
                features_to_remove.add(feat1)
                features_to_keep.add(feat2)
            else:
                features_to_remove.add(feat2)
                features_to_keep.add(feat1)
    
    print(f"✓ Selected {len(features_to_remove)} features to remove")
    print(f"✓ Keeping {len(feature_cols) - len(features_to_remove)} features")
    
    return features_to_remove, features_to_keep


def create_residual_features(df, high_corr_pairs, feature_cols):
    """
    Create residual features by regressing correlated features.
    
    For each pair (X1, X2), create residual: X2_residual = X2 - predict(X2 | X1)
    
    Args:
        df: DataFrame with features
        high_corr_pairs: List of (feature1, feature2, correlation)
        feature_cols: All feature column names
    
    Returns:
        df_residual: DataFrame with residual features added
        residual_info: Dict with residual feature metadata
    """
    print("\n🔧 Creating residual features...")
    
    df_residual = df.copy()
    residual_info = {}
    
    # Track which features have been processed
    processed = set()
    
    for feat1, feat2, corr_val in high_corr_pairs:
        # Skip if already processed
        if feat2 in processed:
            continue
        
        # Regress feat2 on feat1
        X = df[[feat1]].values
        y = df[feat2].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Get predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Replace feat2 with residuals
        residual_col_name = f"{feat2}_residual"
        df_residual[feat2] = residuals
        
        # Store metadata
        residual_info[feat2] = {
            'regressed_on': feat1,
            'correlation': float(corr_val),
            'r_squared': float(model.score(X, y)),
            'residual_std': float(np.std(residuals))
        }
        
        processed.add(feat2)
    
    print(f"✓ Created {len(residual_info)} residual features")
    
    return df_residual, residual_info


def plot_correlation_heatmap(corr_matrix, high_corr_pairs, output_path, threshold=0.995):
    """
    Plot heatmap of highly correlated features.
    
    Args:
        corr_matrix: Full correlation matrix
        high_corr_pairs: List of high correlation pairs
        output_path: Path to save plot
        threshold: Correlation threshold
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️  matplotlib/seaborn not installed, skipping heatmap plot")
        return
    
    if len(high_corr_pairs) == 0:
        print("⚠️  No high correlation pairs to plot")
        return
    
    # Get unique features from high correlation pairs
    unique_features = set()
    for feat1, feat2, _ in high_corr_pairs[:50]:  # Limit to top 50 pairs
        unique_features.add(feat1)
        unique_features.add(feat2)
    
    unique_features = sorted(list(unique_features))
    
    if len(unique_features) > 100:
        print(f"⚠️  Too many features ({len(unique_features)}) to plot clearly")
        unique_features = unique_features[:100]
    
    # Extract subset of correlation matrix
    subset_corr = corr_matrix.loc[unique_features, unique_features]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(subset_corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Heatmap of Features with |r| > {threshold}\n(Top {len(unique_features)} features)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Filter highly correlated features')
    parser.add_argument('--threshold', type=float, default=0.995,
                       help='Correlation threshold (default: 0.995)')
    parser.add_argument('--method', type=str, default='remove',
                       choices=['remove', 'residual'],
                       help='Filtering method: remove or residual (default: remove)')
    parser.add_argument('--input', type=str, default='train_fe.parquet',
                       help='Input data file (default: train_fe.parquet)')
    parser.add_argument('--output-suffix', type=str, default='filtered',
                       help='Suffix for output files (default: filtered)')
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'correlation_filtering'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("FEATURE CORRELATION FILTERING")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"  Correlation threshold: {args.threshold}")
    print(f"  Filtering method: {args.method}")
    print(f"  Input file: {args.input}")
    
    # Load data
    input_path = data_dir / args.input
    if not input_path.exists():
        print(f"\n❌ Error: {input_path} not found")
        return
    
    df = load_data(input_path)
    print(f"\n✓ Loaded data: {df.shape[0]:,} samples, {df.shape[1]} features")
    
    # Identify X1-X780 features (original raw features)
    feature_cols = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
    feature_cols = sorted(feature_cols, key=lambda x: int(x[1:]))  # Sort by number
    
    print(f"✓ Identified {len(feature_cols)} raw features (X1-X{len(feature_cols)})")
    
    if len(feature_cols) == 0:
        print("❌ Error: No X1-X780 features found in dataset")
        return
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df, feature_cols)
    
    # Find high correlation pairs
    high_corr_pairs = find_high_correlation_pairs(corr_matrix, threshold=args.threshold)
    
    if len(high_corr_pairs) == 0:
        print(f"\n✓ No feature pairs with |correlation| > {args.threshold}")
        print("✓ No filtering needed!")
        return
    
    # Print top correlated pairs
    print(f"\n📊 Top 20 most correlated feature pairs:")
    print(f"{'Feature 1':<15} {'Feature 2':<15} {'Correlation':>12}")
    print("-" * 45)
    for feat1, feat2, corr_val in high_corr_pairs[:20]:
        print(f"{feat1:<15} {feat2:<15} {corr_val:>12.6f}")
    
    # Save correlation pairs to CSV
    pairs_df = pd.DataFrame(high_corr_pairs, columns=['feature_1', 'feature_2', 'correlation'])
    pairs_path = results_dir / 'high_correlation_pairs.csv'
    pairs_df.to_csv(pairs_path, index=False)
    print(f"\n✓ Saved all {len(high_corr_pairs)} pairs to {pairs_path}")
    
    # Plot correlation heatmap
    plot_path = results_dir / 'correlation_heatmap.png'
    plot_correlation_heatmap(corr_matrix, high_corr_pairs, plot_path, args.threshold)
    
    # Apply filtering method
    if args.method == 'remove':
        print("\n" + "="*60)
        print("METHOD: REMOVE REDUNDANT FEATURES")
        print("="*60)
        
        features_to_remove, features_to_keep = select_features_to_remove(high_corr_pairs, feature_cols)
        
        # Create filtered dataset
        cols_to_drop = list(features_to_remove)
        df_filtered = df.drop(columns=cols_to_drop)
        
        print(f"\n✓ Removed {len(cols_to_drop)} features")
        print(f"✓ Filtered dataset: {df_filtered.shape[0]:,} samples, {df_filtered.shape[1]} features")
        
        # Save filtered dataset
        output_path = data_dir / f"{args.input.replace('.parquet', '')}_{args.output_suffix}.parquet"
        df_filtered.to_parquet(output_path, index=False)
        print(f"✓ Saved filtered dataset to {output_path}")
        
        # Save removal info
        removal_info = {
            'threshold': args.threshold,
            'method': 'remove',
            'n_pairs': len(high_corr_pairs),
            'n_removed': len(features_to_remove),
            'n_remaining': len(feature_cols) - len(features_to_remove),
            'removed_features': sorted(list(features_to_remove)),
            'kept_features': sorted([f for f in feature_cols if f not in features_to_remove])
        }
        
        removal_info_path = results_dir / 'removal_info.json'
        with open(removal_info_path, 'w') as f:
            json.dump(removal_info, f, indent=2)
        print(f"✓ Saved removal info to {removal_info_path}")
    
    elif args.method == 'residual':
        print("\n" + "="*60)
        print("METHOD: CREATE RESIDUAL FEATURES")
        print("="*60)
        
        df_residual, residual_info = create_residual_features(df, high_corr_pairs, feature_cols)
        
        print(f"\n✓ Created {len(residual_info)} residual features")
        print(f"✓ Residual dataset: {df_residual.shape[0]:,} samples, {df_residual.shape[1]} features")
        
        # Save residual dataset
        output_path = data_dir / f"{args.input.replace('.parquet', '')}_{args.output_suffix}.parquet"
        df_residual.to_parquet(output_path, index=False)
        print(f"✓ Saved residual dataset to {output_path}")
        
        # Save residual info
        residual_metadata = {
            'threshold': args.threshold,
            'method': 'residual',
            'n_pairs': len(high_corr_pairs),
            'n_residuals': len(residual_info),
            'residuals': residual_info
        }
        
        residual_info_path = results_dir / 'residual_info.json'
        with open(residual_info_path, 'w') as f:
            json.dump(residual_metadata, f, indent=2)
        print(f"✓ Saved residual info to {residual_info_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original features (X1-X{len(feature_cols)}): {len(feature_cols)}")
    print(f"High correlation pairs (|r| > {args.threshold}): {len(high_corr_pairs)}")
    
    if args.method == 'remove':
        print(f"Features removed: {len(features_to_remove)}")
        print(f"Features remaining: {len(feature_cols) - len(features_to_remove)}")
        print(f"Reduction: {len(features_to_remove)/len(feature_cols)*100:.1f}%")
    else:
        print(f"Residual features created: {len(residual_info)}")
        print(f"Total features: {df_residual.shape[1]}")
    
    print(f"\n📁 Results saved to: {results_dir}")
    print("\n✓ Correlation filtering completed!")


if __name__ == '__main__':
    main()
