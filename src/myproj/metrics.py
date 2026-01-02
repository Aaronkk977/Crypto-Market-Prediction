"""
Evaluation metrics and performance tracking.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict

def calculate_metrics(y_true, y_pred, prefix="") -> Dict:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "Train", "Validation")
        
    Returns:
        dict: Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # Print metrics
    if prefix:
        print(f"\n{prefix} Set Performance:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")
    
    return metrics

def compare_metrics(metrics_dict: Dict, metric_name='r2') -> None:
    """
    Compare metrics across different experiments.
    
    Args:
        metrics_dict: Dictionary of {experiment_name: metrics_dict}
        metric_name: Metric to compare (default: 'r2')
    """
    print(f"\n{'='*60}")
    print(f"Comparison of {metric_name.upper()}")
    print(f"{'='*60}")
    
    for exp_name, metrics in metrics_dict.items():
        if 'val_metrics' in metrics:
            val_metric = metrics['val_metrics'].get(metric_name, 'N/A')
            print(f"{exp_name:40s}: {val_metric}")
