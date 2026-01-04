#!/usr/bin/env python
"""
Feature Selection Pipeline:
1. Select top N features by feature-label correlation
2. Add hourly time grouping
3. Save filtered dataset for training

Usage:
    python scripts/select_top_features_and_train.py --top_n 200
    python scripts/select_top_features_and_train.py --top_n 120 --input train.parquet --output train_top120_grouped.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data


def load_top_correlated_features(correlation_file, available_features, top_n=200):
    """
    Load top N features by absolute correlation with label.
    
    Args:
        correlation_file: Path to feature_label_correlations.csv
        available_features: Set of available features in dataset
        top_n: Number of top features to select
    
    Returns:
        selected_features: List of feature names
    """
    print(f"\n📊 Loading feature correlations from {correlation_file.name}...")
    
    corr_df = pd.read_csv(correlation_file)
    
    # Filter to only features that exist in the dataset
    corr_df = corr_df[corr_df['feature'].isin(available_features)].reset_index(drop=True)
    
    # Select top N by absolute correlation
    n_available = min(top_n, len(corr_df))
    top_features = corr_df.head(n_available)['feature'].tolist()
    
    print(f"✓ Selected top {len(top_features)} features by |correlation|")
    if len(corr_df) > 0:
        print(f"  Correlation range: {corr_df.iloc[0]['abs_correlation']:.6f} to {corr_df.iloc[n_available-1]['abs_correlation']:.6f}")
    
    return top_features


def add_time_grouping(df, group_size_hours=1):
    """
    Add hourly time grouping to dataset.
    
    Args:
        df: DataFrame with datetime index or date_id/time_id columns
        group_size_hours: Size of each time group in hours (default: 1)
    
    Returns:
        df: DataFrame with time_group_id column added
    """
    if 'time_group_id' in df.columns:
        print(f"\n⏰ time_group_id already exists ({df['time_group_id'].nunique()} groups)")
        return df
    
    print(f"\n⏰ Adding time_group_id (grouping by {group_size_hours} hour(s))...")
    
    if isinstance(df.index, pd.DatetimeIndex):
        # Use datetime index to create hourly groups
        start_time = df.index.min()
        hours_from_start = (df.index - start_time).total_seconds() / 3600
        df['time_group_id'] = (hours_from_start / group_size_hours).astype(int)
    elif 'date_id' in df.columns and 'time_id' in df.columns:
        # Use date_id and time_id if available (assuming time_id is in minutes)
        df['time_group_id'] = df['date_id'] * 24 + (df['time_id'] // (60 * group_size_hours))
    else:
        # Fallback: sequential grouping
        print("  Warning: Using sequential grouping (60 rows per group)")
        df['time_group_id'] = np.arange(len(df)) // 60
    
    n_groups = df['time_group_id'].nunique()
    print(f"  Created {n_groups} time-based groups")
    
    # Show group statistics
    group_sizes = df.groupby('time_group_id').size()
    print(f"  Group size: min={group_sizes.min()}, max={group_sizes.max()}, mean={group_sizes.mean():.1f}")
    
    return df


def save_filtered_dataset(df, selected_features, output_path):
    """
    Save dataset with only selected features.
    
    Args:
        df: Original dataframe
        selected_features: List of feature names to keep
        output_path: Path to save filtered dataset
    """
    print(f"\n💾 Saving filtered dataset to {output_path.name}...")
    
    # Keep selected features + metadata columns + label
    metadata_cols = ['symbol', 'date_id', 'time_id', 'time_group_id']
    label_cols = ['label', 'Y']
    
    cols_to_keep = []
    
    # Add metadata columns that exist
    for col in metadata_cols:
        if col in df.columns:
            cols_to_keep.append(col)
    
    # Add selected features
    cols_to_keep.extend(selected_features)
    
    # Add label column if exists
    for col in label_cols:
        if col in df.columns and col not in cols_to_keep:
            cols_to_keep.append(col)
    
    df_filtered = df[cols_to_keep].copy()
    
    print(f"  Original shape: {df.shape}")
    print(f"  Filtered shape: {df_filtered.shape}")
    print(f"  Features: {len(selected_features)}")
    
    df_filtered.to_parquet(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description='Feature Selection with Time Grouping')
    parser.add_argument('--top_n', type=int, default=200,
                        help='Number of top features to select by correlation (default: 200)')
    parser.add_argument('--input', type=str, default='train.parquet',
                        help='Input data file (default: train.parquet)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name (default: train_top<N>_grouped.parquet)')
    parser.add_argument('--group_hours', type=int, default=1,
                        help='Time group size in hours (default: 1)')
    
    args = parser.parse_args()
    
    # Set up paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'feature_selection'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine output filename
    if args.output is None:
        args.output = f'train_top{args.top_n}_grouped.parquet'
    
    print("\n" + "="*70)
    print("FEATURE SELECTION WITH TIME GROUPING")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Top N features: {args.top_n}")
    print(f"Time grouping: {args.group_hours} hour(s)")
    print("="*70)
    
    # Check correlation file exists
    correlation_file = project_dir / 'results' / 'feature_label_correlation' / 'feature_label_correlations.csv'
    
    if not correlation_file.exists():
        print(f"\n❌ Error: {correlation_file} not found!")
        print("Please run scripts/feature_label_correlation.py first.")
        return
    
    # Load data
    data_path = data_dir / args.input
    print(f"\n📂 Loading data from {data_path.name}...")
    df = load_data(data_path)
    
    # Add time grouping
    df = add_time_grouping(df, group_size_hours=args.group_hours)
    
    # Get available features (exclude metadata and label columns)
    metadata_cols = ['symbol', 'date_id', 'time_id', 'time_group_id']
    label_cols = ['label', 'Y']
    exclude_cols = set(metadata_cols + label_cols)
    available_features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n  Total features in dataset: {len(available_features)}")
    
    # Load top correlated features
    top_features = load_top_correlated_features(correlation_file, available_features, top_n=args.top_n)
    
    if len(top_features) == 0:
        print("\n❌ Error: No features selected!")
        return
    
    # Save filtered dataset
    output_path = data_dir / args.output
    save_filtered_dataset(df, top_features, output_path)
    
    # Save feature list
    feature_list_path = results_dir / f'selected_features_top{len(top_features)}.txt'
    with open(feature_list_path, 'w') as f:
        f.write(f"# Top {len(top_features)} features selected by correlation\n")
        f.write(f"# Source: {args.input}\n")
        f.write(f"# Output: {args.output}\n\n")
        for i, feat in enumerate(top_features, 1):
            f.write(f"{feat}\n")
    
    print(f"\n💾 Saved feature list to {feature_list_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Selected {len(top_features)} features by correlation")
    print(f"✓ Added time_group_id with {df['time_group_id'].nunique()} groups")
    print(f"✓ Saved filtered dataset: {output_path.name}")
    print(f"\nNext steps:")
    print(f"1. Train model: python scripts/train_lightgbm.py")
    print(f"2. Calculate importance: python scripts/permutation_importance.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
