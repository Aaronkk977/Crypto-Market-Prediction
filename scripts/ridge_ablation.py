import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from pathlib import Path
import sys
import joblib
import time
import json

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from split import split_data

"""
Ablation Study Experiments for Ridge Regression Model.

This script runs 4 experiments:
- Experiment A: Only 780 original features (baseline)
- Experiment B: Only new engineered features (signal check)
- Experiment C: 780 + new features (expected improvement)
- Experiment D: Permutation importance analysis
"""

def get_feature_groups(df):
    """
    Identify different feature groups in the dataframe.
    
    Returns:
        original_features: List of X1-X780 features
        engineered_features: List of newly created features
        base_features: bid_qty, ask_qty, buy_qty, sell_qty, volume
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

def train_and_evaluate(X_train, X_val, y_train, y_val, experiment_name, alphas=None):
    """
    Train RidgeCV model and evaluate performance.
    
    Args:
        X_train, X_val, y_train, y_val: Split datasets
        experiment_name: Name of the experiment
        alphas: Alpha values for RidgeCV (default: [0.1, 1.0, 10.0, 100.0])
    
    Returns:
        Dictionary with metrics and model
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]
    
    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print(f"{'='*60}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train RidgeCV with cross-validation (use 3-fold to speed up)
    print(f"Training RidgeCV with alphas: {alphas}...")
    print("Using 3-fold CV for efficiency...")
    start_time = time.time()
    
    model = RidgeCV(alphas=alphas, cv=3, scoring='r2')  # Changed from 5 to 3
    model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    print(f"Best alpha: {model.alpha_}")
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    results = {
        'experiment': experiment_name,
        'n_features': X_train.shape[1],
        'best_alpha': float(model.alpha_),
        'training_time': training_time,
        'train_metrics': {
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        },
        'val_metrics': {
            'mse': mean_squared_error(y_val, y_val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'mae': mean_absolute_error(y_val, y_val_pred),
            'r2': r2_score(y_val, y_val_pred)
        }
    }
    
    # Print results
    print("\nTraining Set:")
    print(f"  MSE:  {results['train_metrics']['mse']:.6f}")
    print(f"  RMSE: {results['train_metrics']['rmse']:.6f}")
    print(f"  MAE:  {results['train_metrics']['mae']:.6f}")
    print(f"  R²:   {results['train_metrics']['r2']:.6f}")
    
    print("\nValidation Set:")
    print(f"  MSE:  {results['val_metrics']['mse']:.6f}")
    print(f"  RMSE: {results['val_metrics']['rmse']:.6f}")
    print(f"  MAE:  {results['val_metrics']['mae']:.6f}")
    print(f"  R²:   {results['val_metrics']['r2']:.6f}")
    
    return results, model, scaler, X_train_scaled, X_val_scaled

def experiment_a(df, split_method='random', test_size=0.2, random_state=42):
    """
    Experiment A: Only 780 original features (baseline)
    """
    original_features, _, _ = get_feature_groups(df)
    
    # Select only original features + label
    df_exp = df[original_features + ['label']]
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        df_exp, split_method=split_method, test_size=test_size, random_state=random_state
    )
    
    # Train and evaluate
    results, model, scaler, _, _ = train_and_evaluate(
        X_train, X_val, y_train, y_val,
        "Experiment A: Only 780 Original Features (Baseline)"
    )
    
    return results

def experiment_b(df, split_method='random', test_size=0.2, random_state=42):
    """
    Experiment B: Only new engineered features (signal check)
    """
    _, engineered_features, base_features = get_feature_groups(df)
    
    # Select only engineered features + base features + label
    # Include base features as they are needed for context
    df_exp = df[base_features + engineered_features + ['label']]
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(
        df_exp, split_method=split_method, test_size=test_size, random_state=random_state
    )
    
    # Train and evaluate
    results, model, scaler, _, _ = train_and_evaluate(
        X_train, X_val, y_train, y_val,
        "Experiment B: Only New Engineered Features"
    )
    
    return results

def experiment_c(df, split_method='random', test_size=0.2, random_state=42):
    """
    Experiment C: 780 + new features (expected improvement)
    """
    # Use all features
    X_train, X_val, y_train, y_val = split_data(
        df, split_method=split_method, test_size=test_size, random_state=random_state
    )
    
    # Train and evaluate
    results, model, scaler, X_train_scaled, X_val_scaled = train_and_evaluate(
        X_train, X_val, y_train, y_val,
        "Experiment C: 780 + New Features (Full Model)"
    )
    
    return results, model, scaler, X_train, X_val, y_val, X_val_scaled

def experiment_d(model, scaler, X_val, y_val, X_val_scaled, df):
    """
    Experiment D: Permutation importance analysis
    """
    print(f"\n{'='*60}")
    print("Experiment D: Permutation Importance Analysis")
    print(f"{'='*60}")
    
    _, engineered_features, _ = get_feature_groups(df)
    feature_names = X_val.columns.tolist()
    
    print("\nCalculating permutation importance on validation set...")
    print("(This may take a few minutes...)")
    
    start_time = time.time()
    perm_importance = permutation_importance(
        model, X_val_scaled, y_val,
        n_repeats=10,
        random_state=42,
        scoring='r2'
    )
    calc_time = time.time() - start_time
    print(f"Completed in {calc_time:.2f}s")
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Check importance of engineered features
    print("\n" + "="*60)
    print("Top 20 Most Important Features:")
    print("="*60)
    for idx, row in importance_df.head(20).iterrows():
        is_new = "✓" if row['feature'] in engineered_features else " "
        print(f"[{is_new}] {row['feature']:30s}  {row['importance_mean']:8.6f} ± {row['importance_std']:.6f}")
    
    print("\n" + "="*60)
    print("Importance of Engineered Features:")
    print("="*60)
    engineered_importance = importance_df[importance_df['feature'].isin(engineered_features)]
    
    if len(engineered_importance) > 0:
        for idx, row in engineered_importance.iterrows():
            print(f"{row['feature']:30s}  {row['importance_mean']:8.6f} ± {row['importance_std']:.6f}")
        
        avg_importance = engineered_importance['importance_mean'].mean()
        print(f"\nAverage importance of engineered features: {avg_importance:.6f}")
        
        near_zero_count = (engineered_importance['importance_mean'].abs() < 0.0001).sum()
        print(f"Features with near-zero importance (<0.0001): {near_zero_count}/{len(engineered_importance)}")
    
    results = {
        'experiment': 'Experiment D: Permutation Importance',
        'calculation_time': calc_time,
        'top_features': importance_df.head(20).to_dict('records'),
        'engineered_features_importance': engineered_importance.to_dict('records')
    }
    
    return results, importance_df

def save_results(results_list, output_dir):
    """Save experiment results to JSON and CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save summary as JSON
    summary_path = output_dir / 'ablation_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    
    # Create comparison table
    comparison = []
    for result in results_list[:3]:  # A, B, C only
        comparison.append({
            'Experiment': result['experiment'],
            'Features': result['n_features'],
            'Best Alpha': result['best_alpha'],
            'Train R²': result['train_metrics']['r2'],
            'Val R²': result['val_metrics']['r2'],
            'Val RMSE': result['val_metrics']['rmse'],
            'Val MAE': result['val_metrics']['mae'],
            'Time (s)': result['training_time']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_path = output_dir / 'ablation_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to {comparison_path}")
    
    return comparison_df

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'ablation'
    
    print("\n" + "="*60)
    print("ABLATION STUDY: RIDGE REGRESSION")
    print("="*60)
    
    # Load feature-engineered data
    train_data_path = data_dir / 'train_fe.parquet'
    print(f"\nLoading data from {train_data_path}...")
    df = pd.read_parquet(train_data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Identify feature groups
    original_features, engineered_features, base_features = get_feature_groups(df)
    print(f"\nFeature Groups:")
    print(f"  Original features (X1-X780): {len(original_features)}")
    print(f"  Base features: {len(base_features)}")
    print(f"  Engineered features: {len(engineered_features)}")
    print(f"  Engineered features: {engineered_features}")
    
    # Run experiments
    results_list = []
    
    # Experiment A: Baseline
    print("\n" + "="*60)
    print("Running Experiment A...")
    print("="*60)
    result_a = experiment_a(df, split_method='random', test_size=0.2, random_state=42)
    results_list.append(result_a)
    
    # Experiment B: Only new features
    print("\n" + "="*60)
    print("Running Experiment B...")
    print("="*60)
    result_b = experiment_b(df, split_method='random', test_size=0.2, random_state=42)
    results_list.append(result_b)
    
    # Experiment C: Full model
    print("\n" + "="*60)
    print("Running Experiment C...")
    print("="*60)
    result_c, model_c, scaler_c, X_train_c, X_val_c, y_val_c, X_val_scaled_c = experiment_c(
        df, split_method='random', test_size=0.2, random_state=42
    )
    results_list.append(result_c)
    
    # Experiment D: Permutation importance
    print("\n" + "="*60)
    print("Running Experiment D...")
    print("="*60)
    result_d, importance_df = experiment_d(model_c, scaler_c, X_val_c, y_val_c, X_val_scaled_c, df)
    results_list.append(result_d)
    
    # Save importance dataframe
    results_dir.mkdir(exist_ok=True, parents=True)
    importance_df.to_csv(results_dir / 'feature_importance.csv', index=False)
    
    # Save all results
    comparison_df = save_results(results_list, results_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Calculate improvements
    baseline_r2 = result_a['val_metrics']['r2']
    full_r2 = result_c['val_metrics']['r2']
    improvement = full_r2 - baseline_r2
    
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    print(f"Baseline (A) Val R²: {baseline_r2:.6f}")
    print(f"Full Model (C) Val R²: {full_r2:.6f}")
    print(f"Improvement: {improvement:+.6f} ({improvement/baseline_r2*100:+.2f}%)")
    
    if improvement > 0.001:
        print("\n✓ Engineered features provide measurable improvement!")
    elif improvement > 0:
        print("\n→ Small improvement detected (may not be significant)")
    else:
        print("\n✗ No improvement from engineered features")
    
    print("\n" + "="*60)
    print("Ablation study completed!")
    print("="*60)

if __name__ == "__main__":
    main()
