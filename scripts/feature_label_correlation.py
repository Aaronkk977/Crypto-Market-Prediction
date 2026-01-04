#!/usr/bin/env python
"""
Feature-Label Correlation Analysis for Crypto Market Prediction.

This script:
1. Calculates Pearson correlation between each feature (X1-X780) and the target label (Y)
2. Ranks features by absolute correlation with Y
3. Identifies features with:
   - Strong positive/negative correlation (potential predictors)
   - Weak correlation (candidates for removal)
4. Saves correlation analysis results and visualizations

Usage:
    python scripts/feature_label_correlation.py
    python scripts/feature_label_correlation.py --top_n 50
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data


def calculate_feature_label_correlations(df, feature_cols, target_col='label'):
    """
    Calculate Pearson correlation between each feature and the target label.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column (default: 'label')
    
    Returns:
        correlations_df: DataFrame with feature, correlation, p_value
    """
    print(f"\n📊 Calculating correlations between {len(feature_cols)} features and '{target_col}'...")
    
    correlations = []
    
    for i, feature in enumerate(feature_cols):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(feature_cols)} features processed...")
        
        # Calculate Pearson correlation and p-value
        corr, p_value = pearsonr(df[feature], df[target_col])
        
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_value
        })
    
    # Create DataFrame and sort by absolute correlation
    correlations_df = pd.DataFrame(correlations)
    correlations_df = correlations_df.sort_values('abs_correlation', ascending=False)
    
    print(f"✓ Correlation analysis completed for {len(feature_cols)} features")
    
    return correlations_df


def analyze_correlation_distribution(correlations_df):
    """
    Analyze the distribution of correlations.
    
    Args:
        correlations_df: DataFrame with correlation results
    
    Returns:
        stats: Dictionary with distribution statistics
    """
    print("\n📈 Analyzing correlation distribution...")
    
    corr_values = correlations_df['correlation'].values
    abs_corr_values = correlations_df['abs_correlation'].values
    
    stats = {
        'mean_correlation': float(np.mean(corr_values)),
        'std_correlation': float(np.std(corr_values)),
        'mean_abs_correlation': float(np.mean(abs_corr_values)),
        'max_positive_correlation': float(np.max(corr_values)),
        'max_negative_correlation': float(np.min(corr_values)),
        'max_abs_correlation': float(np.max(abs_corr_values)),
        'median_abs_correlation': float(np.median(abs_corr_values)),
        'num_features': len(correlations_df),
    }
    
    # Count features by correlation strength
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
    for threshold in thresholds:
        count = (abs_corr_values >= threshold).sum()
        stats[f'num_abs_corr_gte_{threshold}'] = int(count)
    
    print(f"\n  Mean correlation: {stats['mean_correlation']:.6f}")
    print(f"  Mean |correlation|: {stats['mean_abs_correlation']:.6f}")
    print(f"  Max positive: {stats['max_positive_correlation']:.6f}")
    print(f"  Max negative: {stats['max_negative_correlation']:.6f}")
    print(f"  Max |correlation|: {stats['max_abs_correlation']:.6f}")
    
    print("\n  Features by correlation strength:")
    for threshold in thresholds:
        count = stats[f'num_abs_corr_gte_{threshold}']
        pct = 100 * count / stats['num_features']
        print(f"    |corr| ≥ {threshold}: {count} ({pct:.1f}%)")
    
    return stats


def plot_correlation_distribution(correlations_df, output_dir):
    """
    Create visualizations of correlation distribution.
    
    Args:
        correlations_df: DataFrame with correlation results
        output_dir: Directory to save plots
    """
    print("\n📊 Creating correlation visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram of correlations
    ax = axes[0, 0]
    ax.hist(correlations_df['correlation'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    ax.set_xlabel('Correlation with Y', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Feature-Label Correlations', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram of absolute correlations
    ax = axes[0, 1]
    ax.hist(correlations_df['abs_correlation'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('|Correlation| with Y', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Absolute Correlations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Top 30 features by absolute correlation
    ax = axes[1, 0]
    top_30 = correlations_df.head(30)
    colors = ['green' if c > 0 else 'red' for c in top_30['correlation']]
    bars = ax.barh(range(len(top_30)), top_30['correlation'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(top_30['feature'], fontsize=8)
    ax.set_xlabel('Correlation with Y', fontsize=12)
    ax.set_title('Top 30 Features by |Correlation| with Y', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_abs_corr = np.sort(correlations_df['abs_correlation'].values)[::-1]
    ax.plot(range(len(sorted_abs_corr)), sorted_abs_corr, linewidth=2)
    ax.set_xlabel('Feature Rank', fontsize=12)
    ax.set_ylabel('|Correlation| with Y', fontsize=12)
    ax.set_title('Features Ranked by |Correlation|', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines for reference
    for threshold in [0.1, 0.2, 0.3]:
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(sorted_abs_corr) * 0.7, threshold + 0.01, f'{threshold}', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / 'feature_label_correlations.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def identify_weak_features(correlations_df, threshold=0.01):
    """
    Identify features with very weak correlation to the label.
    
    Args:
        correlations_df: DataFrame with correlation results
        threshold: Absolute correlation threshold (default: 0.01)
    
    Returns:
        weak_features: List of features with |correlation| < threshold
    """
    print(f"\n🔍 Identifying features with |correlation| < {threshold}...")
    
    weak_features = correlations_df[
        correlations_df['abs_correlation'] < threshold
    ]['feature'].tolist()
    
    print(f"✓ Found {len(weak_features)} features with |correlation| < {threshold}")
    
    return weak_features


def main():
    parser = argparse.ArgumentParser(description='Feature-Label Correlation Analysis')
    parser.add_argument('--top_n', type=int, default=50,
                        help='Number of top features to display (default: 50)')
    parser.add_argument('--weak_threshold', type=float, default=0.01,
                        help='Threshold for weak correlation (default: 0.01)')
    parser.add_argument('--data_file', type=str, default='train.parquet',
                        help='Data file to use (default: train.parquet)')
    
    args = parser.parse_args()
    
    # Set up paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    output_dir = project_dir / 'results' / 'feature_label_correlation'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("FEATURE-LABEL CORRELATION ANALYSIS")
    print("="*70)
    
    # Load data
    data_path = data_dir / args.data_file
    if not data_path.exists():
        print(f"\n⚠️  {args.data_file} not found, trying train.parquet...")
        data_path = data_dir / 'train.parquet'
    
    print(f"\n📂 Loading data from {data_path.name}...")
    df = load_data(data_path)
    
    # Get feature columns (X1-X780)
    feature_cols = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
    print(f"✓ Found {len(feature_cols)} feature columns")
    
    # Check target column
    target_col = 'label' if 'label' in df.columns else 'Y'
    if target_col not in df.columns:
        print(f"\n❌ Error: Target column '{target_col}' not found in dataset!")
        return
    
    print(f"✓ Target column '{target_col}' found with {df[target_col].notna().sum()} non-null values")
    
    # Calculate correlations
    start_time = pd.Timestamp.now()
    correlations_df = calculate_feature_label_correlations(df, feature_cols, target_col=target_col)
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    print(f"⏱️  Correlation calculation took {elapsed:.2f} seconds")
    
    # Analyze distribution
    stats = analyze_correlation_distribution(correlations_df)
    
    # Save full results
    correlations_path = output_dir / 'feature_label_correlations.csv'
    correlations_df.to_csv(correlations_path, index=False)
    print(f"\n💾 Saved full correlation results to {correlations_path}")
    
    # Save statistics
    stats_path = output_dir / 'correlation_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Saved statistics to {stats_path}")
    
    # Display top N features
    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} FEATURES BY ABSOLUTE CORRELATION WITH {target_col.upper()}")
    print(f"{'='*70}")
    print(f"\n{'Rank':<6} {'Feature':<12} {'Correlation':<15} {'|Correlation|':<15} {'P-value'}")
    print("-" * 70)
    
    for idx, row in correlations_df.head(args.top_n).iterrows():
        rank = correlations_df.index.get_loc(idx) + 1
        sign = "+" if row['correlation'] > 0 else ""
        print(f"{rank:<6} {row['feature']:<12} {sign}{row['correlation']:<14.6f} "
              f"{row['abs_correlation']:<14.6f} {row['p_value']:.2e}")
    
    # Identify weak features
    weak_features = identify_weak_features(correlations_df, threshold=args.weak_threshold)
    
    if weak_features:
        weak_path = output_dir / 'weak_correlation_features.txt'
        with open(weak_path, 'w') as f:
            f.write(f"# Features with |correlation| < {args.weak_threshold}\n")
            f.write(f"# Total: {len(weak_features)} features\n\n")
            for feat in weak_features:
                corr_val = correlations_df[correlations_df['feature'] == feat]['correlation'].iloc[0]
                f.write(f"{feat}\t{corr_val:.6f}\n")
        
        print(f"\n💾 Saved weak features list to {weak_path}")
        
        print(f"\n📋 Sample of weak features (first 10):")
        for feat in weak_features[:10]:
            corr_val = correlations_df[correlations_df['feature'] == feat]['correlation'].iloc[0]
            print(f"  {feat}: {corr_val:.6f}")
        
        if len(weak_features) > 10:
            print(f"  ... and {len(weak_features) - 10} more")
    
    # Create visualizations
    plot_correlation_distribution(correlations_df, output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total features analyzed: {len(feature_cols)}")
    print(f"Mean |correlation|: {stats['mean_abs_correlation']:.6f}")
    print(f"Strongest correlation: {stats['max_abs_correlation']:.6f}")
    print(f"Features with |corr| ≥ 0.1: {stats['num_abs_corr_gte_0.1']}")
    print(f"Features with |corr| < {args.weak_threshold}: {len(weak_features)}")
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
