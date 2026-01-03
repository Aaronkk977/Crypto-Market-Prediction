#!/usr/bin/env python
"""
Check data structure for repetitive patterns that may require GroupKFold.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_data_structure():
    """Analyze the data to identify repetitive structures."""
    
    data_path = Path('data/train_fe.parquet')
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    print("\n" + "="*60)
    print("DATA STRUCTURE ANALYSIS")
    print("="*60)
    
    df = pd.read_parquet(data_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Total samples: {df.shape[0]:,}")
    print(f"Total features: {df.shape[1]}")
    
    print("\n" + "="*60)
    print("COLUMN NAMES")
    print("="*60)
    print(list(df.columns))
    
    print("\n" + "="*60)
    print("FIRST 10 ROWS")
    print("="*60)
    print(df.head(10))
    
    print("\n" + "="*60)
    print("CHECKING FOR REPETITIVE STRUCTURES")
    print("="*60)
    
    # Check for potential grouping columns
    potential_group_cols = []
    
    for col in df.columns:
        nunique = df[col].nunique()
        pct_unique = nunique / len(df) * 100
        
        # Potential group column if:
        # 1. Much fewer unique values than total rows
        # 2. Not the target column
        # 3. Contains identifiable patterns (id, symbol, asset, stock, date, time)
        
        if pct_unique < 50 and col != 'label':
            potential_group_cols.append((col, nunique, pct_unique))
    
    print(f"\nFound {len(potential_group_cols)} potential grouping columns:")
    print(f"{'Column':<30} {'Unique Values':<15} {'% Unique':<10}")
    print("-" * 60)
    
    for col, nunique, pct in sorted(potential_group_cols, key=lambda x: x[2]):
        print(f"{col:<30} {nunique:<15,} {pct:<10.2f}%")
    
    # Check for time-series patterns
    print("\n" + "="*60)
    print("CHECKING FOR TIME-SERIES PATTERNS")
    print("="*60)
    
    # Look for columns with names suggesting time/sequence
    time_cols = [col for col in df.columns if any(
        keyword in col.lower() 
        for keyword in ['time', 'date', 'timestamp', 'row_id', 'id', 'symbol', 'stock', 'asset']
    )]
    
    if time_cols:
        print(f"\nFound columns with time/id-related names: {time_cols}")
        for col in time_cols:
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].nunique():,}")
            print(f"  Sample values: {df[col].head(10).tolist()}")
    else:
        print("\nNo obvious time/id columns found in column names")
    
    # Check for repeated patterns in features
    print("\n" + "="*60)
    print("CHECKING FOR DUPLICATED FEATURE PATTERNS")
    print("="*60)
    
    # Drop label column for this analysis
    feature_cols = [col for col in df.columns if col != 'label']
    X = df[feature_cols]
    
    # Check for duplicate rows
    n_duplicates = X.duplicated().sum()
    pct_duplicates = n_duplicates / len(X) * 100
    
    print(f"\nDuplicated feature rows: {n_duplicates:,} ({pct_duplicates:.2f}%)")
    
    if n_duplicates > 0:
        print("⚠️  WARNING: Found duplicate feature rows!")
        print("This suggests high repetitive structure - GroupKFold recommended!")
    
    # Check if data has sequential structure
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if potential_group_cols:
        print("\n✓ Found potential grouping columns for GroupKFold:")
        for col, nunique, pct in potential_group_cols[:5]:
            print(f"  - {col} ({nunique:,} groups)")
        print("\n➤ RECOMMENDATION: Implement GroupKFold to avoid label leakage")
    
    if n_duplicates > len(df) * 0.01:  # More than 1% duplicates
        print("\n⚠️  High level of duplicated patterns detected")
        print("➤ RECOMMENDATION: GroupKFold is CRITICAL to prevent overfitting")
    
    # Check if there's a pattern suggesting crypto market data
    if any('symbol' in col.lower() for col in df.columns):
        print("\n➤ Detected crypto/stock symbol column")
        print("➤ RECOMMENDATION: Use symbol-based GroupKFold (same symbol should stay together)")
    
    if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
        print("\n➤ Detected time-based columns")
        print("➤ RECOMMENDATION: Consider TimeSeriesSplit or GroupKFold by time periods")

if __name__ == '__main__':
    analyze_data_structure()
