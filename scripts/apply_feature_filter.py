#!/usr/bin/env python
"""
Apply pre-computed feature filtering to test dataset.

This script applies the same feature removal as computed on training data
to ensure train/test consistency.

Usage:
    python scripts/apply_feature_filter.py --input test_fe.parquet
    python scripts/apply_feature_filter.py --input test_fe_grouped.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import json
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data


def main():
    parser = argparse.ArgumentParser(description='Apply feature filtering to dataset')
    parser.add_argument('--input', type=str, required=True,
                       help='Input data file (e.g., test_fe.parquet)')
    parser.add_argument('--filter-info', type=str, 
                       default='results/correlation_filtering/removal_info.json',
                       help='Path to removal_info.json from training')
    parser.add_argument('--output-suffix', type=str, default='filtered',
                       help='Suffix for output file (default: filtered)')
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    filter_info_path = project_dir / args.filter_info
    
    print("\n" + "="*60)
    print("APPLY FEATURE FILTERING")
    print("="*60)
    
    # Load filter info
    if not filter_info_path.exists():
        print(f"\n❌ Error: {filter_info_path} not found")
        print("   Run: python scripts/filter_correlated_features.py first")
        return
    
    with open(filter_info_path, 'r') as f:
        filter_info = json.load(f)
    
    removed_features = filter_info['removed_features']
    
    print(f"\n📋 Filter configuration:")
    print(f"  Threshold: {filter_info['threshold']}")
    print(f"  Method: {filter_info['method']}")
    print(f"  Features to remove: {len(removed_features)}")
    
    # Load data
    input_path = data_dir / args.input
    if not input_path.exists():
        print(f"\n❌ Error: {input_path} not found")
        return
    
    print(f"\n📂 Loading {args.input}...")
    df = load_data(input_path)
    print(f"✓ Loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
    
    # Check which features exist in dataset
    existing_to_remove = [f for f in removed_features if f in df.columns]
    missing_features = [f for f in removed_features if f not in df.columns]
    
    print(f"\n🔍 Feature check:")
    print(f"  Features to remove (exist): {len(existing_to_remove)}")
    if missing_features:
        print(f"  ⚠️  Features not found: {len(missing_features)}")
        print(f"      {missing_features[:10]}...")
    
    # Apply filtering
    df_filtered = df.drop(columns=existing_to_remove)
    
    print(f"\n✓ Removed {len(existing_to_remove)} features")
    print(f"✓ Filtered dataset: {df_filtered.shape[0]:,} samples, {df_filtered.shape[1]} features")
    
    # Save filtered dataset
    output_path = data_dir / f"{args.input.replace('.parquet', '')}_{args.output_suffix}.parquet"
    df_filtered.to_parquet(output_path, index=False)
    print(f"✓ Saved filtered dataset to {output_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input: {args.input} ({df.shape[1]} features)")
    print(f"Output: {output_path.name} ({df_filtered.shape[1]} features)")
    print(f"Reduction: {len(existing_to_remove)} features ({len(existing_to_remove)/df.shape[1]*100:.1f}%)")
    print("\n✓ Filtering completed!")


if __name__ == '__main__':
    main()
