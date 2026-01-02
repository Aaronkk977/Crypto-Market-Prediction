import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

"""
Data Splitting Module.
Provides functions to split data into training and validation sets.
Supports both random split and time-based split strategies.
"""

def random_split(X, y, test_size=0.2, random_state=42):
    """
    Split data randomly into train and validation sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    print(f"\nPerforming random split (test_size={test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_val, y_train, y_val

def time_split(X, y, test_size=0.2):
    """
    Split data by time order (no shuffling).
    Uses the last test_size proportion as validation set.
    Assumes data is already sorted by time.
    
    Args:
        X: Feature matrix (assumed to be time-ordered)
        y: Target vector (assumed to be time-ordered)
        test_size: Proportion of data to use for validation (default: 0.2)
    
    Returns:
        X_train, X_val, y_train, y_val
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
               target_col='label', drop_cols=None):
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
        X_train, X_val, y_train, y_val
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

def load_split(split_dir):
    """
    Load previously saved split data.
    
    Args:
        split_dir: Directory containing split data
    
    Returns:
        X_train, X_val, y_train, y_val
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

def main():
    """
    Example usage of data splitting module.
    """
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    split_dir = data_dir / 'splits'
    
    # Load data
    print("="*60)
    print("DATA SPLITTING EXAMPLE")
    print("="*60)
    
    train_data_path = data_dir / 'train_fe.parquet'
    df = pd.read_parquet(train_data_path)
    print(f"\nLoaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Perform random split
    X_train, X_val, y_train, y_val = split_data(
        df, 
        split_method='random',  # Change to 'time' for time-based split
        test_size=0.2,
        random_state=42
    )
    
    # Save split
    save_split(X_train, X_val, y_train, y_val, split_dir)
    
    # Load split (to verify)
    X_train_loaded, X_val_loaded, y_train_loaded, y_val_loaded = load_split(split_dir)
    
    print("\n" + "="*60)
    print("Data splitting completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
