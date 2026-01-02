"""
Data loading and validation utilities.
"""

import pandas as pd
from pathlib import Path
from typing import Union

def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from parquet file with validation.
    
    Args:
        data_path: Path to parquet file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.suffix != '.parquet':
        raise ValueError(f"Expected .parquet file, got: {data_path.suffix}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

def validate_dataframe(df: pd.DataFrame, required_cols: list = None) -> bool:
    """
    Validate dataframe structure.
    
    Args:
        df: Dataframe to validate
        required_cols: List of required column names
        
    Returns:
        bool: True if valid
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return True

def get_feature_groups(df: pd.DataFrame) -> tuple:
    """
    Identify different feature groups in the dataframe.
    
    Returns:
        tuple: (original_features, engineered_features, base_features)
    """
    all_cols = df.columns.tolist()
    
    # Original anonymized features (X1-X780)
    original_features = [c for c in all_cols if c.startswith('X')]
    
    # Base market features
    base_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    
    # Engineered features (everything else except label)
    engineered_features = [c for c in all_cols 
                          if c not in original_features 
                          and c not in base_features 
                          and c != 'label']
    
    return original_features, engineered_features, base_features
