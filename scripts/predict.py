#!/usr/bin/env python
"""
Generate predictions using trained model.

Usage:
    python scripts/predict.py                                    # Use default model (ridge_model.pkl)
    python scripts/predict.py --model lightgbm_model.txt         # Use LightGBM model
    python scripts/predict.py --model ridge_model.pkl --scaler scaler.pkl
    python scripts/predict.py --model artifacts/models/elasticnet_model.pkl
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.utils import load_model, load_scaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_top_features(feature_importance_path, top_n=200):
    """Load top N features from feature importance CSV file."""
    if not Path(feature_importance_path).exists():
        print(f"Warning: Feature importance file not found at {feature_importance_path}")
        return None
    
    df_importance = pd.read_csv(feature_importance_path)
    top_features = df_importance.head(top_n)['feature'].tolist()
    return top_features

def filter_test_features(df_test, top_features):
    """Filter test data to only include top features."""
    if top_features is None:
        return df_test
    
    # Keep only features that exist in the dataframe
    available_features = [f for f in top_features if f in df_test.columns]
    missing_features = [f for f in top_features if f not in df_test.columns]
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in test data")
    
    df_filtered = df_test[available_features]
    print(f"Filtered test data: {len(available_features)} features (from {len(df_test.columns)} original features)")
    return df_filtered

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate predictions using trained model')
    parser.add_argument('--model', type=str, default='lightgbm_model.txt',
                        help='Model filename or path (default: lightgbm_model.txt)')
    parser.add_argument('--scaler', type=str, default='scaler.pkl',
                        help='Scaler filename or path (default: scaler.pkl)')
    parser.add_argument('--test-data', type=str, default='test_fe_filtered.parquet',
                        help='Test data filename (default: test_fe_filtered.parquet)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: predictions_{model_name}.csv)')
    parser.add_argument('--config', type=str, default='configs/lightgbm.yaml',
                        help='Path to config file for feature selection (default: configs/lightgbm.yaml)')
    args = parser.parse_args()
    
    # Set up paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    model_dir = project_dir / 'artifacts' / 'models'
    scaler_dir = project_dir / 'artifacts' / 'scalers'
    output_dir = project_dir / 'artifacts' / 'predictions'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("PREDICTION PIPELINE")
    print("="*60)
    
    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = model_dir / args.model
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    print(f"Model: {model_path}")
    
    # Load configuration for feature selection
    config_path = project_dir / args.config
    config = load_config(config_path) if config_path.exists() else {}
    
    # Load test data
    test_path = data_dir / args.test_data
    df_test = load_data(test_path)
    
    # Remove label column if it exists (test set has dummy labels)
    if 'label' in df_test.columns:
        df_test = df_test.drop(columns=['label'])
    
    # Apply feature selection if enabled in config
    if config.get('feature_selection', {}).get('enabled', False):
        top_n = config['feature_selection']['top_n_features']
        importance_file = config['feature_selection']['importance_file']
        feature_importance_path = project_dir / importance_file
        print(f"\nFeature selection enabled: Using top {top_n} features")
        top_features = load_top_features(feature_importance_path, top_n)
        df_test = filter_test_features(df_test, top_features)
    
    # Determine model type
    model_type = 'lightgbm' if model_path.suffix == '.txt' else 'sklearn'
    
    # Load model
    if model_type == 'lightgbm':
        if not HAS_LIGHTGBM:
            print("❌ Error: LightGBM is not installed")
            print("Install with: pip install lightgbm")
            return
        print(f"\nLoading LightGBM model from {model_path}...")
        model = lgb.Booster(model_file=str(model_path))
        need_scaler = False
    else:
        print(f"\nLoading sklearn model from {model_path}...")
        model = load_model(model_path)
        need_scaler = True
    
    # Scale features and make predictions
    if need_scaler:
        # Resolve scaler path
        scaler_path = Path(args.scaler)
        if not scaler_path.is_absolute():
            scaler_path = scaler_dir / args.scaler
        
        if not scaler_path.exists():
            print(f"❌ Error: Scaler not found at {scaler_path}")
            return
        
        print(f"Scaler: {scaler_path}")
        scaler = load_scaler(scaler_path)
        
        print("\nScaling features...")
        X_test_scaled = scaler.transform(df_test)
        
        print("Generating predictions...")
        predictions = model.predict(X_test_scaled)
    else:
        # LightGBM doesn't need scaling
        print("\nGenerating predictions (no scaling needed)...")
        predictions = model.predict(df_test)
    
    # Prepare output filename
    if args.output:
        output_filename = args.output
    else:
        model_name = model_path.stem
        output_filename = f'predictions_{model_name}.csv'
    
    output_path = output_dir / output_filename
    
    # Create DataFrame with correct format
    pred_df = pd.DataFrame({
        'ID': np.arange(1, len(predictions) + 1),
        'prediction': predictions
    })
    
    # Save predictions
    pred_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Model type: {model_type}")
    print(f"Output: {output_path}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std:  {predictions.std():.6f}")
    print(f"  Min:  {predictions.min():.6f}")
    print(f"  Max:  {predictions.max():.6f}")
    
    print("\n" + "="*60)
    print("Prediction pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()