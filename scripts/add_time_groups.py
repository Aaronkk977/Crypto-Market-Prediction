#!/usr/bin/env python
"""
Add time-based group IDs to the dataset for GroupKFold.

For time-series data, we create groups based on time periods to ensure
temporal dependencies are respected and prevent label leakage.

Usage:
    python scripts/add_time_groups.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_time_groups(input_path, output_path, group_size_minutes=60):
    """
    Add time-based group IDs to data.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save output with group_id column
        group_size_minutes: Size of each time group in minutes (default: 60)
    """
    print("\n" + "="*60)
    print("ADDING TIME-BASED GROUP IDS")
    print("="*60)
    
    print(f"\nLoading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Reset index to get datetime as a column
    print(f"\nOriginal index type: {type(df.index)}")
    
    # If index is numeric, load original data to get proper datetime
    if not isinstance(df.index, pd.DatetimeIndex) or (isinstance(df.index, pd.DatetimeIndex) and df.index.min().year == 1970):
        print("\n⚠️  Warning: Index is not DatetimeIndex or appears corrupted")
        print("Loading original data to recover datetime index...")
        
        # Load original data
        orig_path = str(input_path).replace('_fe.parquet', '.parquet')
        if Path(orig_path).exists():
            df_orig = pd.read_parquet(orig_path)
            if isinstance(df_orig.index, pd.DatetimeIndex) and df_orig.index.min().year >= 2000:
                print(f"✓ Recovered datetime index from {orig_path}")
                df.index = df_orig.index
            else:
                print("⚠️  Original data doesn't have datetime index (likely test data)")
                print("Creating sequential group IDs based on row order...")
                df['time_group_id'] = np.arange(len(df)) // group_size_minutes
                n_groups = df['time_group_id'].nunique()
                print(f"Created {n_groups} sequential groups (size={group_size_minutes} rows each)")
                
                # Save and return
                print(f"\nSaving data with group IDs to {output_path}...")
                df.to_parquet(output_path)
                print("✓ Saved successfully!")
                return df
        else:
            print(f"❌ Original file not found: {orig_path}")
            print("Creating sequential group IDs based on row order...")
            df['time_group_id'] = np.arange(len(df)) // group_size_minutes
            n_groups = df['time_group_id'].nunique()
            print(f"Created {n_groups} sequential groups (size={group_size_minutes} rows each)")
            return df
    
    print(f"✓ Data has valid DatetimeIndex")
    print(f"  Start: {df.index.min()}")
    print(f"  End: {df.index.max()}")
    print(f"  Duration: {df.index.max() - df.index.min()}")
    
    # Create time-based groups
    # Group by time periods (e.g., hourly, daily)
    print(f"\nCreating time groups (period={group_size_minutes} minutes)...")
    
    # Calculate minutes since start
    start_time = df.index.min()
    minutes_from_start = (df.index - start_time).total_seconds() / 60
    
    # Assign group IDs
    df['time_group_id'] = (minutes_from_start // group_size_minutes).astype(int)
    
    n_groups = df['time_group_id'].nunique()
    print(f"Created {n_groups} time-based groups")
    
    # Show group statistics
    group_sizes = df.groupby('time_group_id').size()
    print(f"\nGroup size statistics:")
    print(f"  Mean: {group_sizes.mean():.1f} samples")
    print(f"  Median: {group_sizes.median():.1f} samples")
    print(f"  Min: {group_sizes.min()} samples")
    print(f"  Max: {group_sizes.max()} samples")
    
    # Sample a few groups to show
    print(f"\nSample groups:")
    for gid in df['time_group_id'].unique()[:5]:
        group_df = df[df['time_group_id'] == gid]
        print(f"  Group {gid}: {len(group_df)} samples, "
              f"time range: {group_df.index.min()} to {group_df.index.max()}")
    
    print(f"\nSaving data with group IDs to {output_path}...")
    df.to_parquet(output_path)
    print("✓ Saved successfully!")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR GROUPKFOLD")
    print("="*60)
    print(f"\nUse the 'time_group_id' column for GroupKFold:")
    print(f"  - Number of groups: {n_groups}")
    print(f"  - Recommended n_splits: {min(5, n_groups // 2)}")
    print(f"\nExample usage:")
    print(f"  split_method='group_kfold'")
    print(f"  group_col='time_group_id'")
    print(f"  n_splits={min(5, n_groups // 2)}")
    
    return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Add time-based group IDs to dataset')
    parser.add_argument('--input', type=str, default='train_fe.parquet',
                       help='Input file name (default: train_fe.parquet)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name (default: <input>_grouped.parquet)')
    parser.add_argument('--group-size', type=int, default=60,
                       help='Group size in minutes (default: 60)')
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    
    # Determine input and output paths
    input_path = data_dir / args.input
    if args.output:
        output_path = data_dir / args.output
    else:
        # Auto-generate output name
        output_path = data_dir / args.input.replace('.parquet', '_grouped.parquet')
    
    if input_path.exists():
        print(f"\n📊 Processing {args.input}...")
        df = add_time_groups(input_path, output_path, group_size_minutes=args.group_size)
    else:
        print(f"❌ File not found: {input_path}")
        return
    
    print("\n" + "="*60)
    print("✓ GROUPING COMPLETED")
    print("="*60)
    print(f"\nOutput: {output_path}")
    print("\nNext steps:")
    print("1. Use the grouped file for training with GroupKFold")
    print("2. Update training scripts to use:")
    print("   - split_method='group_kfold'")
    print("   - group_col='time_group_id'")

if __name__ == '__main__':
    main()
