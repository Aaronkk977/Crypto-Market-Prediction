"""
Model training and prediction utilities.
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import time
from typing import Tuple, Optional, Dict

import lightgbm as lgb



def train_ridge_model(X_train, y_train, alpha=1.0, **kwargs):
    """
    Train Ridge Regression model with specified alpha.
    
    Args:
        X_train: Training features (should be scaled)
        y_train: Training target
        alpha: Regularization strength
        **kwargs: Additional arguments for Ridge model
        
    Returns:
        Trained Ridge model
    """
    print(f"\nTraining Ridge Regression (alpha={alpha})...")
    start_time = time.time()
    
    model = Ridge(alpha=alpha, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

def train_ridge_cv(X_train, y_train, alphas=None, cv=3, scoring='pearson', **kwargs):
    """
    Train RidgeCV model with cross-validation for alpha selection.
    
    Args:
        X_train: Training features (should be scaled)
        y_train: Training target
        alphas: List of alpha values to try (default: [0.1, 1.0, 10.0, 100.0])
        cv: Number of cross-validation folds (default: 3)
        scoring: Scoring metric ('pearson', 'r2', or custom scorer). Default: 'pearson'
        **kwargs: Additional arguments for RidgeCV model
        
    Returns:
        Trained RidgeCV model
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]
    
    # Handle scoring
    if scoring == 'pearson':
        from .metrics import get_pearson_scorer
        scoring = get_pearson_scorer()
    
    print(f"\nTraining RidgeCV with alphas: {alphas}...")
    print(f"Using {cv}-fold cross-validation...")
    start_time = time.time()
    
    model = RidgeCV(alphas=alphas, cv=cv, scoring=scoring, **kwargs)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best alpha: {model.alpha_}")
    
    return model

def train_elasticnet_cv(X_train, y_train, l1_ratio=0.5, alphas=None, cv=3, n_jobs=4, **kwargs):
    """
    Train ElasticNetCV model with cross-validation for alpha selection.
    
    Args:
        X_train: Training features (should be scaled)
        y_train: Training target
        l1_ratio: The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
                  l1_ratio=0: Ridge (L2), l1_ratio=1: Lasso (L1)
        alphas: Array of alpha values to try (default: 50 values from 0.001 to 1000)
        cv: Number of cross-validation folds (default: 3)
        n_jobs: Number of parallel jobs (default: 4)
        **kwargs: Additional arguments for ElasticNetCV model
        
    Returns:
        Trained ElasticNetCV model
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    
    print(f"\nTraining ElasticNetCV (L1 ratio={l1_ratio})...")
    print(f"Using {cv}-fold cross-validation with {len(alphas)} alpha values...")
    start_time = time.time()
    
    model = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        cv=cv,
        max_iter=100000,  # Increased from 10000 to 100000
        tol=1e-4,  # Default tolerance
        random_state=42,
        n_jobs=n_jobs,
        **kwargs
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    n_nonzero = np.sum(model.coef_ != 0)
    sparsity = (1 - n_nonzero / len(model.coef_)) * 100
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best alpha: {model.alpha_:.6f}")
    print(f"Non-zero coefficients: {n_nonzero}/{len(model.coef_)} ({sparsity:.1f}% sparse)")
    
    return model

def train_elasticnet_sweep(X_train, y_train, l1_ratios=None, alphas=None, cv=3, n_jobs=4):
    """
    Train multiple ElasticNet models with different L1 ratios.
    
    Args:
        X_train: Training features (should be scaled)
        y_train: Training target
        l1_ratios: List of L1 ratios to try (default: [0.05, 0.1, 0.2, 0.5, 0.8])
        alphas: Array of alpha values to try
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary mapping l1_ratio to trained model
    """
    if l1_ratios is None:
        l1_ratios = [0.05, 0.1, 0.2, 0.5, 0.8]
    
    print(f"\n{'='*60}")
    print(f"ELASTIC NET L1 RATIO SWEEP")
    print(f"{'='*60}")
    print(f"L1 ratios to test: {l1_ratios}")
    
    models = {}
    for l1_ratio in l1_ratios:
        model = train_elasticnet_cv(
            X_train, y_train,
            l1_ratio=l1_ratio,
            alphas=alphas,
            cv=cv,
            n_jobs=n_jobs
        )
        models[l1_ratio] = model
    
    return models

def scale_features(X_train, X_val=None, scaler=None) -> Tuple:
    """
    Standardize features to have mean=0 and std=1.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        scaler: Pre-fitted scaler (optional, for transform only)
        
    Returns:
        Tuple: (X_train_scaled, X_val_scaled, scaler) or (X_train_scaled, None, scaler)
    """
    if scaler is None:
        print("\nFitting and transforming features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        print("\nTransforming features with existing scaler...")
        X_train_scaled = scaler.transform(X_train)
    
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    
    print("Features scaled to mean=0, std=1")
    
    return X_train_scaled, X_val_scaled, scaler

def evaluate_model(model, X_train, y_train, X_val, y_val, metrics_module=None):
    """
    Evaluate model performance on train and validation sets.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        metrics_module: Module with calculate_metrics function
        
    Returns:
        dict: Dictionary containing metrics
    """
    from .metrics import calculate_metrics
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred, prefix="Train")
    
    # Validation set predictions
    y_val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred, prefix="Validation")
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_predictions': y_train_pred,
        'val_predictions': y_val_pred
    }

def pearson_eval(y_pred, train_data):
    """
    Custom Pearson correlation evaluation metric for LightGBM.
    
    Args:
        y_pred: Predicted values
        train_data: LightGBM Dataset object
        
    Returns:
        Tuple of (metric_name, metric_value, is_higher_better)
    """
    y_true = train_data.get_label()
    
    # Calculate Pearson correlation
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = np.sqrt(np.sum((y_true - mean_true)**2) * np.sum((y_pred - mean_pred)**2))
    
    if denominator == 0:
        pearson = 0.0
    else:
        pearson = numerator / denominator
    
    # Return: (metric_name, metric_value, is_higher_better)
    return 'pearson', pearson, True

def train_lightgbm(X_train, y_train, X_val, y_val, params: Dict, num_boost_round=1000, 
                   early_stopping_rounds=50, min_delta=0.0):
    """
    Train LightGBM model with early stopping based on Pearson correlation.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: LightGBM parameters dictionary
        num_boost_round: Maximum number of boosting iterations
        early_stopping_rounds: Stop if no improvement for this many rounds
        min_delta: Minimum improvement required to consider as progress (default: 0.0)
        
    Returns:
        Trained LightGBM model, metrics dict
    """
    
    print(f"\nTraining LightGBM...")
    print(f"Parameters: {params}")
    if min_delta > 0:
        print(f"Early stopping: {early_stopping_rounds} rounds, min_delta={min_delta}")
    start_time = time.time()
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model with custom Pearson metric and callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, min_delta=min_delta),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        feval=pearson_eval,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score}")
    
    # Calculate metrics once and return them
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    train_pearson = np.corrcoef(y_train, y_train_pred)[0, 1]
    
    val_rmse = np.sqrt(np.mean((y_val - y_val_pred)**2))
    val_mae = np.mean(np.abs(y_val - y_val_pred))
    val_pearson = np.corrcoef(y_val, y_val_pred)[0, 1]
    
    metrics = {
        'train': {'pearson': train_pearson, 'rmse': train_rmse, 'mae': train_mae},
        'val': {'pearson': val_pearson, 'rmse': val_rmse, 'mae': val_mae}
    }
    
    return model, metrics
