"""
Data splitting utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from pathlib import Path
from typing import Tuple, Generator, Optional

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

def kfold_split(X, y, n_splits=5, random_state=42, shuffle=True) -> Generator:
    """
    Generate K-Fold cross-validation splits.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_splits: Number of folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
        shuffle: Whether to shuffle data before splitting (default: True)
    
    Yields:
        Tuple: (fold_idx, X_train, X_val, y_train, y_val) for each fold
    """
    print(f"\nPerforming {n_splits}-Fold Cross-Validation...")
    print(f"Shuffle: {shuffle}, Random state: {random_state}")
    
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]
        
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
        else:
            y_train = y[train_idx]
            y_val = y[val_idx]
        
        print(f"\nFold {fold_idx}/{n_splits}:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        
        yield fold_idx, X_train, X_val, y_train, y_val

def group_kfold_split(X, y, groups, n_splits=5) -> Generator:
    """
    Generate GroupKFold cross-validation splits.
    Ensures samples from the same group are kept together in train/val.
    Critical for time-series data or data with repetitive structures.
    
    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels for each sample (e.g., time_id, stock_id)
        n_splits: Number of folds (default: 5)
    
    Yields:
        Tuple: (fold_idx, X_train, X_val, y_train, y_val) for each fold
    """
    print(f"\n⚠️  Using GroupKFold Cross-Validation (prevents label leakage)")
    print(f"Number of splits: {n_splits}")
    print(f"Number of groups: {len(np.unique(groups))}")
    print(f"Total samples: {len(X)}")
    
    gkf = GroupKFold(n_splits=n_splits)
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]
        
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
        else:
            y_train = y[train_idx]
            y_val = y[val_idx]
        
        # Get group info for this fold
        train_groups = groups[train_idx] if isinstance(groups, np.ndarray) else groups.iloc[train_idx]
        val_groups = groups[val_idx] if isinstance(groups, np.ndarray) else groups.iloc[val_idx]
        n_train_groups = len(np.unique(train_groups))
        n_val_groups = len(np.unique(val_groups))
        
        print(f"\nFold {fold_idx}/{n_splits}:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%) from {n_train_groups} groups")
        print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%) from {n_val_groups} groups")
        
        # Verify no group leakage
        train_groups_set = set(np.unique(train_groups))
        val_groups_set = set(np.unique(val_groups))
        overlap = train_groups_set & val_groups_set
        if len(overlap) > 0:
            print(f"  ⚠️  WARNING: {len(overlap)} groups overlap between train and val!")
        else:
            print(f"  ✓ No group overlap (label leakage prevented)")
        
        yield fold_idx, X_train, X_val, y_train, y_val

def split_data(df, split_method='random', test_size=0.2, random_state=42, 
               target_col='label', drop_cols=None, n_splits=5, 
               group_col: Optional[str] = None) -> Tuple:
    """
    Split dataframe into features and target, then into train/validation sets.
    
    Args:
        df: Input dataframe
        split_method: 'random', 'time', 'kfold', or 'group_kfold' (default: 'random')
        test_size: Proportion for validation set (default: 0.2) - not used for kfold
        random_state: Random seed for random split (default: 42)
        target_col: Name of target column (default: 'label')
        drop_cols: Additional columns to drop from features (default: None)
        n_splits: Number of folds for kfold/group_kfold (default: 5)
        group_col: Column name for groups in GroupKFold (required for 'group_kfold')
    
    Returns:
        Tuple: (X_train, X_val, y_train, y_val) for single split
        Generator: yields (fold_idx, X_train, X_val, y_train, y_val) for kfold
    """
    print("\nPreparing data for splitting...")
    
    # Extract groups if using GroupKFold
    groups = None
    if split_method == 'group_kfold':
        if group_col is None:
            raise ValueError("group_col must be specified for group_kfold split method")
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in dataframe")
        groups = df[group_col].values
        print(f"Using GroupKFold with group column: {group_col}")
        print(f"Number of unique groups: {len(np.unique(groups))}")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Drop additional columns if specified (including group_col)
    cols_to_drop = list(drop_cols) if drop_cols else []
    if group_col and group_col in X.columns:
        cols_to_drop.append(group_col)
    
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop, errors='ignore')
        print(f"Dropped columns: {cols_to_drop}")
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Perform split based on method
    if split_method == 'random':
        return random_split(X, y, test_size=test_size, random_state=random_state)
    elif split_method == 'time':
        return time_split(X, y, test_size=test_size)
    elif split_method == 'kfold':
        return kfold_split(X, y, n_splits=n_splits, random_state=random_state)
    elif split_method == 'group_kfold':
        return group_kfold_split(X, y, groups, n_splits=n_splits)
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Use 'random', 'time', 'kfold', or 'group_kfold'.")

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
