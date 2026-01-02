#!/usr/bin/env python
"""
Generate engineered features from raw data.

Usage:
    python scripts/make_features.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.features import add_depth_features
import time

def main():
    # Set up paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    
    # Input and output paths
    train_input = data_dir / 'train.parquet'
    test_input = data_dir / 'test.parquet'
    train_output = data_dir / 'train_fe.parquet'
    test_output = data_dir / 'test_fe.parquet'
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Process training data
    print(f"\n{'='*60}")
    print("Processing Training Data")
    print(f"{'='*60}")
    
    df_train = load_data(train_input)
    start_time = time.time()
    df_train_fe = add_depth_features(df_train)
    fe_time = time.time() - start_time
    
    print(f"Feature engineering completed in {fe_time:.2f}s")
    print(f"New shape: {df_train_fe.shape}")
    print(f"\nSaving to {train_output}...")
    df_train_fe.to_parquet(train_output, index=False)
    print("Saved successfully!")
    
    # Process test data
    print(f"\n{'='*60}")
    print("Processing Test Data")
    print(f"{'='*60}")
    
    df_test = load_data(test_input)
    start_time = time.time()
    df_test_fe = add_depth_features(df_test)
    fe_time = time.time() - start_time
    
    print(f"Feature engineering completed in {fe_time:.2f}s")
    print(f"New shape: {df_test_fe.shape}")
    print(f"\nSaving to {test_output}...")
    df_test_fe.to_parquet(test_output, index=False)
    print("Saved successfully!")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training data: {df_train_fe.shape}")
    print(f"Test data: {df_test_fe.shape}")
    print(f"\nOutput files:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    print("\n" + "="*60)
    print("Feature engineering completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
