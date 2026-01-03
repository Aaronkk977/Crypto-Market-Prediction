#!/usr/bin/env python
"""
Ablation Study for Ridge Regression Model.

Experiments:
- A: Only 780 original features (baseline)
- B: Only engineered features
- C: All features (full model)
- D: Permutation importance analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data, get_feature_groups
from myproj.split import split_data
from myproj.metrics import calculate_metrics, get_pearson_scorer


def train_and_evaluate(X_train, X_val, y_train, y_val, experiment_name):
    """Train RidgeCV model and evaluate performance."""
    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print(f"{'='*60}")
    print(f"Features: {X_train.shape[1]} | Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train
    print("\nTraining RidgeCV (3-fold CV)...")
    start_time = time.time()
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3, scoring='r2')
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"Completed in {training_time:.2f}s | Best alpha: {model.alpha_}")
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    train_metrics = calculate_metrics(y_train, y_train_pred, prefix="Train")
    val_metrics = calculate_metrics(y_val, y_val_pred, prefix="Validation")
    
    results = {
        'experiment': experiment_name,
        'n_features': X_train.shape[1],
        'best_alpha': float(model.alpha_),
        'training_time': training_time,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    
    return results, model, scaler, X_train_scaled, X_val_scaled


def run_experiment(df, features_to_use, experiment_name):
    """Run a single experiment with specified features."""
    df_exp = df[features_to_use + ['label']]
    X_train, X_val, y_train, y_val = split_data(df_exp, split_method='random', 
                                                  test_size=0.2, random_state=42)
    return train_and_evaluate(X_train, X_val, y_train, y_val, experiment_name)

def analyze_importance(model, X_val, y_val, X_val_scaled, engineered_features):
    """Calculate and analyze permutation importance."""
    print(f"\n{'='*60}")
    print("Experiment D: Permutation Importance Analysis")
    print(f"{'='*60}")
    
    print("\nCalculating permutation importance...")
    print("Using conservative settings to avoid overloading the system...")
    start_time = time.time()
    
    perm_importance = permutation_importance(
        model, X_val_scaled, y_val,
        n_repeats=10, random_state=42, 
        scoring=get_pearson_scorer(), n_jobs=4  # Limit to 4 cores
    )
    
    calc_time = time.time() - start_time
    print(f"Completed in {calc_time:.2f}s")
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': X_val.columns.tolist(),
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'is_engineered': [f in engineered_features for f in X_val.columns]
    }).sort_values('importance_mean', ascending=False)
    
    # Print top features
    print("\nTop 20 Features:")
    for _, row in importance_df.head(20).iterrows():
        marker = "✓" if row['is_engineered'] else " "
        print(f"[{marker}] {row['feature']:30s}  {row['importance_mean']:8.6f} ± {row['importance_std']:.6f}")
    
    # Engineered features summary
    eng_df = importance_df[importance_df['is_engineered']]
    if len(eng_df) > 0:
        print(f"\nEngineered features: {len(eng_df)}")
        print(f"Average importance: {eng_df['importance_mean'].mean():.6f}")
        print(f"Near-zero (<0.0001): {(eng_df['importance_mean'].abs() < 0.0001).sum()}")
    
    return importance_df, calc_time


def save_results(results_list, importance_df, output_dir):
    """Save experiment results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save JSON
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Save comparison table
    comparison = pd.DataFrame([{
        'Experiment': r['experiment'],
        'Features': r['n_features'],
        'Best Alpha': r['best_alpha'],
        'Train R²': r['train_metrics']['r2'],
        'Train Pearson': r['train_metrics']['pearson'],
        'Val R²': r['val_metrics']['r2'],
        'Val Pearson': r['val_metrics']['pearson'],
        'Val RMSE': r['val_metrics']['rmse'],
        'Time (s)': r['training_time']
    } for r in results_list[:3]])  # A, B, C only
    
    comparison.to_csv(output_dir / 'ablation_comparison.csv', index=False)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    return comparison


def main():
    # Setup
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'ablation'
    
    print("\n" + "="*60)
    print("ABLATION STUDY: RIDGE REGRESSION")
    print("="*60)
    
    # Load data
    df = load_data(data_dir / 'train_fe.parquet')
    original_features, engineered_features, base_features = get_feature_groups(df)
    
    print(f"\nFeature Groups:")
    print(f"  Original (X1-X780): {len(original_features)}")
    print(f"  Base: {len(base_features)}")
    print(f"  Engineered: {len(engineered_features)}")
    
    # Run experiments
    results = []
    
    # Experiment A: Baseline (original features only)
    result_a, *_ = run_experiment(df, original_features, 
                                   "Experiment A: Original Features Only (Baseline)")
    results.append(result_a)
    
    # Experiment B: Engineered features only
    result_b, *_ = run_experiment(df, base_features + engineered_features,
                                   "Experiment B: Engineered Features Only")
    results.append(result_b)
    
    # Experiment C: Full model
    X_train, X_val, y_train, y_val = split_data(df, split_method='random',
                                                  test_size=0.2, random_state=42)
    result_c, model_c, scaler_c, _, X_val_scaled_c = train_and_evaluate(
        X_train, X_val, y_train, y_val, "Experiment C: All Features (Full Model)")
    results.append(result_c)
    
    # Experiment D: Permutation importance
    importance_df, calc_time = analyze_importance(model_c, X_val, y_val, 
                                                   X_val_scaled_c, engineered_features)
    
    # Save results
    comparison = save_results(results, importance_df, results_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(comparison.to_string(index=False))
    
    baseline_r2 = result_a['val_metrics']['r2']
    full_r2 = result_c['val_metrics']['r2']
    improvement = full_r2 - baseline_r2
    
    print(f"\nBaseline Val R²: {baseline_r2:.6f}")
    print(f"Full Model Val R²: {full_r2:.6f}")
    print(f"Improvement: {improvement:+.6f} ({improvement/baseline_r2*100:+.2f}%)")
    
    baseline_pearson = result_a['val_metrics']['pearson']
    full_pearson = result_c['val_metrics']['pearson']
    pearson_improvement = full_pearson - baseline_pearson
    
    print(f"\nBaseline Val Pearson: {baseline_pearson:.6f}")
    print(f"Full Model Val Pearson: {full_pearson:.6f}")
    print(f"Improvement: {pearson_improvement:+.6f} ({pearson_improvement/baseline_pearson*100:+.2f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
