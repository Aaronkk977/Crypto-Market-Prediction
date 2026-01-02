import pandas as pd
import os
from pathlib import Path

# Get the script's directory and construct path to data folder
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'

# Read parquet files
train_df = pd.read_parquet(data_dir / 'train.parquet')
test_df = pd.read_parquet(data_dir / 'test.parquet')

print("\n" + "=" * 60)
print("TRAINING DATA ANALYSIS")
print("=" * 60)

# Print basic information
print("\n" + "=" * 60)
print("Basic Dataset Information")
print("=" * 60)
print(f"Total rows: {len(train_df)}")
print(f"Total columns: {len(train_df.columns)}")
print()

# Print column names and data types
print("=" * 60)
print("Column Information")
print("=" * 60)
print(train_df.dtypes)
print()

# Print first 10 rows
print("=" * 60)
print("First 10 Rows")
print("=" * 60)
print(train_df.head(10))
print()

# Print statistical summary (sample first 10000 rows)
print("=" * 60)
print("Statistical Summary (first 10000 rows sample)")
print("=" * 60)
print(train_df.head(10000).describe())
print()

# Check missing values
print("=" * 60)
print("Missing Values Count")
print("=" * 60)
missing_train = train_df.isnull().sum()
total_missing_train = missing_train.sum()
print(f"Total missing values: {total_missing_train}")
if total_missing_train > 0:
    print("\nColumns with missing values:")
    print(missing_train[missing_train > 0].sort_values(ascending=False))
else:
    print("No missing values found in training data!")

print("\n\n" + "=" * 60)
print("TEST DATA ANALYSIS")
print("=" * 60)

# Print basic information
print("\n" + "=" * 60)
print("Basic Dataset Information")
print("=" * 60)
print(f"Total rows: {len(test_df)}")
print(f"Total columns: {len(test_df.columns)}")
print()

# Print column names and data types
print("=" * 60)
print("Column Information")
print("=" * 60)
print(test_df.dtypes)
print()

# Print first 10 rows
print("=" * 60)
print("First 10 Rows")
print("=" * 60)
print(test_df.head(10))
print()

# Print statistical summary (sample first 10000 rows)
print("=" * 60)
print("Statistical Summary (first 10000 rows sample)")
print("=" * 60)
print(test_df.head(10000).describe())
print()

# Check missing values
print("=" * 60)
print("Missing Values Count")
print("=" * 60)
missing_test = test_df.isnull().sum()
total_missing_test = missing_test.sum()
print(f"Total missing values: {total_missing_test}")
if total_missing_test > 0:
    print("\nColumns with missing values:")
    print(missing_test[missing_test > 0].sort_values(ascending=False))
else:
    print("No missing values found in test data!")

# Compare datasets
print("\n\n" + "=" * 60)
print("DATASET COMPARISON")
print("=" * 60)
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Common columns: {len(set(train_df.columns) & set(test_df.columns))}")
print(f"Train-only columns: {set(train_df.columns) - set(test_df.columns)}")
print(f"Test-only columns: {set(test_df.columns) - set(train_df.columns)}")
