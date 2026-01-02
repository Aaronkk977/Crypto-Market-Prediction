import pandas as pd
import numpy as np
from pathlib import Path
import time

"""
Feature Engineering Pipeline.
This script reads training and test datasets from parquet files,
adds order book depth and trade imbalance features, and saves the
enhanced datasets back to parquet files.
"""

EPS = 1e-9

def add_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add order book depth and trade imbalance features.
    Only uses features available in each row (no temporal ordering dependency).
    """
    print("Adding depth and imbalance features...")
    df = df.copy()
    
    # Core features
    df["imbalance_best"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + EPS)
    df["trade_imbalance"] = (df["buy_qty"] - df["sell_qty"]) / (df["buy_qty"] + df["sell_qty"] + EPS)
    df["vol_log1p"] = np.log1p(df["volume"])
    df["bid_qty_log1p"] = np.log1p(df["bid_qty"])
    df["ask_qty_log1p"] = np.log1p(df["ask_qty"])
    
    # Additional features
    total_best_qty = df["bid_qty"] + df["ask_qty"]  # Intermediate calculation
    df["total_best_qty_log1p"] = np.log1p(total_best_qty)
    
    book_to_trade_ratio = total_best_qty / (df["volume"] + EPS)  # Intermediate calculation
    df["book_to_trade_ratio_log1p"] = np.log1p(book_to_trade_ratio)
    
    # Interaction features
    df["bid_to_buy"] = np.log1p(df["bid_qty"]) * (df["buy_qty"] / (df["volume"] + EPS))
    df["ask_to_sell"] = np.log1p(df["ask_qty"]) * (df["sell_qty"] / (df["volume"] + EPS))
    
    # Select only the desired features
    feature_cols = [
        "imbalance_best", "trade_imbalance", "vol_log1p",
        "bid_qty_log1p", "ask_qty_log1p",
        "total_best_qty_log1p",
        "book_to_trade_ratio_log1p",
        "bid_to_buy", "ask_to_sell"
    ]
    
    # Convert to float32 to reduce memory usage
    for col in feature_cols:
        df[col] = df[col].astype("float32")
    
    # Keep only original columns + new features
    original_cols = [c for c in df.columns if c not in feature_cols]
    df = df[original_cols + feature_cols]
    
    print(f"Added {len(feature_cols)} new features")
    return df

def process_dataset(input_path: Path, output_path: Path, dataset_name: str):
    """Process a single dataset and save with engineered features"""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading {input_path}...")
    start_time = time.time()
    df = pd.read_parquet(input_path)
    load_time = time.time() - start_time
    print(f"Loaded in {load_time:.2f}s: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Add features
    start_time = time.time()
    df_fe = add_depth_features(df)
    fe_time = time.time() - start_time
    print(f"Feature engineering completed in {fe_time:.2f}s")
    print(f"New shape: {df_fe.shape[0]} rows, {df_fe.shape[1]} columns")
    
    # Save to parquet
    print(f"Saving to {output_path}...")
    start_time = time.time()
    df_fe.to_parquet(output_path, index=False)
    save_time = time.time() - start_time
    print(f"Saved in {save_time:.2f}s")
    
    # Memory info
    memory_mb = df_fe.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {memory_mb:.2f} MB")
    
    return df_fe.shape

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    
    # Input files
    train_input = data_dir / 'train.parquet'
    test_input = data_dir / 'test.parquet'
    
    # Output files
    train_output = data_dir / 'train_fe.parquet'
    test_output = data_dir / 'test_fe.parquet'
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Process training data
    train_shape = process_dataset(train_input, train_output, "Training Data")
    
    # Process test data
    test_shape = process_dataset(test_input, test_output, "Test Data")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training data: {train_shape}")
    print(f"Test data: {test_shape}")
    print(f"\nOutput files:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    print("\n" + "="*60)
    print("Feature engineering completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
