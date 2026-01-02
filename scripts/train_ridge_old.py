import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import joblib
import time
from ../src/split import split_data

"""
Simple Ridge Regression Training Pipeline.
This script loads training data from a parquet file, splits it into training and validation sets 
(default 80/20 split), trains a Ridge Regression model with feature scaling, 
evaluates its performance, and saves the trained model to disk.
"""

def load_data(data_path):
    """Load training data from parquet file"""
    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_data(df, split_method='random', test_size=0.2, random_state=42, remove_correlated=True):
    """Split data into features and target, then train/validation sets"""
    print("\nPreparing data...")
    
    # Remove highly correlated features to avoid ill-conditioned matrix
    if remove_correlated:
        print("Checking for highly correlated features...")
        # Remove features with correlation > 0.95 with another feature
        features_to_remove = []
        
        # Known highly correlated feature pairs (from analysis)
        if 'imbalance_weighted' in df.columns:
            features_to_remove.append('imbalance_weighted')
        if 'trade_pressure' in df.columns:
            features_to_remove.append('trade_pressure')
        
        if features_to_remove:
            print(f"Removing highly correlated features: {features_to_remove}")
            df = df.drop(columns=features_to_remove)
    
    # Use data_split module for splitting
    X_train, X_val, y_train, y_val = split_data(
        df,
        split_method=split_method,
        test_size=test_size,
        random_state=random_state,
        target_col='label'
    )
    
    return X_train, X_val, y_train, y_val

def scale_features(X_train, X_val):
    """Standardize features to have mean=0 and std=1"""
    print("\nScaling features...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("Features scaled to mean=0, std=1")
    
    return X_train_scaled, X_val_scaled, scaler

def train_ridge_model(X_train, y_train, alpha=1.0):
    """Train Ridge Regression model"""
    print(f"\nTraining Ridge Regression (alpha={alpha})...")
    start_time = time.time()
    
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate model performance on train and validation sets"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Training set performance
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print("\nTraining Set Performance:")
    print(f"  MSE:  {train_mse:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE:  {train_mae:.6f}")
    print(f"  R²:   {train_r2:.6f}")
    
    # Validation set performance
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print("\nValidation Set Performance:")
    print(f"  MSE:  {val_mse:.6f}")
    print(f"  RMSE: {val_rmse:.6f}")
    print(f"  MAE:  {val_mae:.6f}")
    print(f"  R²:   {val_r2:.6f}")
    
    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }

def save_model(model, scaler, model_path, scaler_path):
    """Save trained model and scaler to disk"""
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Model saved successfully!")
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    print("Scaler saved successfully!")

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    model_dir = project_dir / 'models'
    
    # Create models directory if it doesn't exist
    model_dir.mkdir(exist_ok=True)
    
    # Load data
    train_data_path = data_dir / 'train_fe.parquet'
    df = load_data(train_data_path)
    
    # Prepare data (removes highly correlated features and splits)
    # Use split_method='random' for random split or 'time' for time-based split
    X_train, X_val, y_train, y_val = prepare_data(
        df, 
        split_method='random',  # Current: random split
        test_size=0.2, 
        random_state=42
    )
    
    # Scale features (important for Ridge regression!)
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    
    # Train model
    alpha = 1.0  # Ridge regularization parameter
    model = train_ridge_model(X_train_scaled, y_train, alpha=alpha)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Save model and scaler
    model_path = model_dir / 'ridge_model.pkl'
    scaler_path = model_dir / 'scaler.pkl'
    save_model(model, scaler, model_path, scaler_path)
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
