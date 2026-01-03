#!/usr/bin/env python
"""
Generate predictions using trained model.

Usage:
    python scripts/predict.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.utils import load_model, load_scaler

def main():
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
    
    # Load test data
    test_path = data_dir / 'test_fe_filtered.parquet'
    df_test = load_data(test_path)
    
    # Remove label column if it exists (test set has dummy labels)
    if 'label' in df_test.columns:
        df_test = df_test.drop(columns=['label'])
    
    # Load model and scaler
    model = load_model(model_dir / 'ridge_model.pkl')
    scaler = load_scaler(scaler_dir / 'scaler.pkl')
    
    # Scale features
    print("\nScaling features...")
    X_test_scaled = scaler.transform(df_test)
    
    # Make predictions
    print("Generating predictions...")
    predictions = model.predict(X_test_scaled)
    
    # --- 修改部分：按照正確格式儲存預測結果 ---
    output_path = output_dir / 'predictions.csv'
    
    # 建立包含 ID (從 1 開始) 與 prediction 的 DataFrame
    pred_df = pd.DataFrame({
        'ID': np.arange(1, len(predictions) + 1),
        'prediction': predictions
    })
    
    # 儲存為 CSV，不包含 pandas 預設索引
    pred_df.to_csv(output_path, index=False)
    # ---------------------------------------
    
    print(f"\nPredictions saved to {output_path}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std:  {predictions.std():.6f}")
    print(f"  Min:  {predictions.min():.6f}")
    print(f"  Max:  {predictions.max():.6f}")
    
    print("\n" + "="*60)
    print("Prediction pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()