#!/usr/bin/env python
"""
Calculate permutation importance for trained model.

Usage:
    python scripts/permutation_importance.py
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.inspection import permutation_importance

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data, get_feature_groups
from myproj.split import split_data
from myproj.utils import load_model, load_scaler
from myproj.metrics import get_pearson_scorer
import time

def main():
    # Set up paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    model_dir = project_dir / 'artifacts' / 'models'
    scaler_dir = project_dir / 'artifacts' / 'scalers'
    output_dir = project_dir / 'results' / 'importance'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load data
    train_path = data_dir / 'train_fe.parquet'
    df = load_data(train_path)
    
    # Split data (use same split as training)
    _, X_val, _, y_val = split_data(
        df,
        split_method='random',
        test_size=0.2,
        random_state=42
    )
    
    # Load model and scaler
    model = load_model(model_dir / 'ridge_model.pkl')
    scaler = load_scaler(scaler_dir / 'scaler.pkl')
    
    # Scale validation features
    print("\nScaling features...")
    X_val_scaled = scaler.transform(X_val)
    
    # Calculate permutation importance
    print("\nCalculating permutation importance...")
    print("(This may take a few minutes...)")
    start_time = time.time()
    
    # Use Pearson correlation scorer
    pearson_score = get_pearson_scorer()
    
    perm_importance = permutation_importance(
        model, X_val_scaled, y_val,
        n_repeats=10, 
        random_state=42,
        scoring=pearson_score,
        n_jobs=4  # Limit to 4 cores instead of all (-1)
    )
    
    calc_time = time.time() - start_time
    print(f"Completed in {calc_time:.2f}s")
    
    # Create importance dataframe
    feature_names = X_val.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Get feature groups
    _, engineered_features, _ = get_feature_groups(df)
    importance_df['is_engineered'] = importance_df['feature'].isin(engineered_features)
    
    # Save results
    output_path = output_dir / 'permutation_importance.csv'
    importance_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Top 20 Most Important Features:")
    print("="*60)
    for idx, row in importance_df.head(20).iterrows():
        marker = "✓" if row['is_engineered'] else " "
        print(f"[{marker}] {row['feature']:30s}  {row['importance_mean']:8.6f} ± {row['importance_std']:.6f}")
    
    # Engineered features summary
    print("\n" + "="*60)
    print("Engineered Features Summary:")
    print("="*60)
    eng_features = importance_df[importance_df['is_engineered']]
    
    if len(eng_features) > 0:
        print(f"Total engineered features: {len(eng_features)}")
        print(f"Average importance: {eng_features['importance_mean'].mean():.6f}")
        print(f"Near-zero importance (<0.0001): {(eng_features['importance_mean'].abs() < 0.0001).sum()}")
        
        print("\nEngineered features by importance:")
        for idx, row in eng_features.iterrows():
            print(f"  {row['feature']:30s}  {row['importance_mean']:8.6f} ± {row['importance_std']:.6f}")
    
    print("\n" + "="*60)
    print("Permutation importance analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main()
