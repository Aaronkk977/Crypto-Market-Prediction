"""
Data splitting utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple

def random_split(X, y, test_size=0.2, random_state=42) -> Tuple:
    """
    Split data randomly into train and validation sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple: (X_train, X_val, y_train, y_val)
    """
    print(f"\nPerforming random split (test_size={test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_val, y_train, y_val

def time_split(X, y, test_size=0.2) -> Tuple:
    """
    Split data by time order (no shuffling).
    Uses the last test_size proportion as validation set.
    Assumes data is already sorted by time.
    
    Args:
        X: Feature matrix (assumed to be time-ordered)
        y: Target vector (assumed to be time-ordered)
        test_size: Proportion of data to use for validation (default: 0.2)
    
    Returns:
        Tuple: (X_train, X_val, y_train, y_val)
    """
    print(f"\nPerforming time-based split (test_size={test_size})...")
    print("Note: Assuming data is already sorted by time")
    
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx] if isinstance(X, pd.DataFrame) else X[:split_idx]
    X_val = X.iloc[split_idx:] if isinstance(X, pd.DataFrame) else X[split_idx:]
    y_train = y.iloc[:split_idx] if isinstance(y, pd.Series) else y[:split_idx]
    y_val = y.iloc[split_idx:] if isinstance(y, pd.Series) else y[split_idx:]
    
    print(f"Train set: {len(X_train)} samples (first {(1-test_size)*100:.0f}%)")
    print(f"Validation set: {len(X_val)} samples (last {test_size*100:.0f}%)")
    
    return X_train, X_val, y_train, y_val

def split_data(df, split_method='random', test_size=0.2, random_state=42, 
               target_col='label', drop_cols=None) -> Tuple:
    """
    Split dataframe into features and target, then into train/validation sets.
    
    Args:
        df: Input dataframe
        split_method: 'random' or 'time' (default: 'random')
        test_size: Proportion for validation set (default: 0.2)
        random_state: Random seed for random split (default: 42)
        target_col: Name of target column (default: 'label')
        drop_cols: Additional columns to drop from features (default: None)
    
    Returns:
        Tuple: (X_train, X_val, y_train, y_val)
    """
    print("\nPreparing data for splitting...")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Drop additional columns if specified
    if drop_cols:
        X = X.drop(columns=drop_cols, errors='ignore')
        print(f"Dropped columns: {drop_cols}")
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Perform split based on method
    if split_method == 'random':
        return random_split(X, y, test_size=test_size, random_state=random_state)
    elif split_method == 'time':
        return time_split(X, y, test_size=test_size)
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Use 'random' or 'time'.")

def save_split(X_train, X_val, y_train, y_val, output_dir):
    """
    Save split data to disk for reuse.
    
    Args:
        X_train, X_val, y_train, y_val: Split datasets
        output_dir: Directory to save split data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSaving split data to {output_dir}...")
    
    # Save as parquet for efficiency
    pd.DataFrame(X_train).to_parquet(output_dir / 'X_train.parquet', index=False)
    pd.DataFrame(X_val).to_parquet(output_dir / 'X_val.parquet', index=False)
    pd.Series(y_train).to_frame('label').to_parquet(output_dir / 'y_train.parquet', index=False)
    pd.Series(y_val).to_frame('label').to_parquet(output_dir / 'y_val.parquet', index=False)
    
    print("Split data saved successfully!")

def load_split(split_dir) -> Tuple:
    """
    Load previously saved split data.
    
    Args:
        split_dir: Directory containing split data
    
    Returns:
        Tuple: (X_train, X_val, y_train, y_val)
    """
    split_dir = Path(split_dir)
    
    print(f"\nLoading split data from {split_dir}...")
    
    X_train = pd.read_parquet(split_dir / 'X_train.parquet')
    X_val = pd.read_parquet(split_dir / 'X_val.parquet')
    y_train = pd.read_parquet(split_dir / 'y_train.parquet')['label']
    y_val = pd.read_parquet(split_dir / 'y_val.parquet')['label']
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val
