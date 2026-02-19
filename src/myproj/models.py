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


# ---------------------------------------------------------------------------
#  XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val,
                  params: Optional[Dict] = None,
                  seeds: list = None,
                  num_boost_round: int = 2000,
                  early_stopping_rounds: int = 50):
    """
    Train XGBoost regressors with multiple random seeds for ensemble.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        params: XGBoost parameters (sensible defaults provided).
        seeds: List of random seeds (default: [42, 123, 456]).
        num_boost_round: Max boosting rounds (default: 2000).
        early_stopping_rounds: Early-stop patience (default: 50).

    Returns:
        list[dict]: One entry per seed with keys
            'seed', 'model', 'best_iteration', 'metrics'.
    """
    import xgboost as xgb

    if seeds is None:
        seeds = [42, 123, 456]

    default_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'verbosity': 0,
    }
    if params:
        default_params.update(params)

    print(f"\n{'='*60}")
    print("XGBOOST MULTI-SEED TRAINING")
    print(f"{'='*60}")
    print(f"Seeds: {seeds}")
    print(f"Params: {default_params}")

    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        default_params['seed'] = seed
        start_time = time.time()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Custom Pearson eval metric for early stopping
        def pearson_eval(predt, dtrain_inner):
            labels = dtrain_inner.get_label()
            corr = float(np.corrcoef(labels, predt)[0, 1])
            return 'pearson', corr

        model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            custom_metric=pearson_eval,
            early_stopping_rounds=early_stopping_rounds,
            maximize=True,
            verbose_eval=100,
        )

        best_iter = model.best_iteration
        y_train_pred = model.predict(dtrain, iteration_range=(0, best_iter + 1))
        y_val_pred = model.predict(dval, iteration_range=(0, best_iter + 1))

        train_pearson = float(np.corrcoef(y_train, y_train_pred)[0, 1])
        val_pearson = float(np.corrcoef(y_val, y_val_pred)[0, 1])

        elapsed = time.time() - start_time
        print(f"  Best iter: {best_iter} | "
              f"Train Pearson: {train_pearson:.6f} | "
              f"Val Pearson: {val_pearson:.6f} | "
              f"Time: {elapsed:.1f}s")

        results.append({
            'seed': seed,
            'model': model,
            'best_iteration': best_iter,
            'y_val_pred': y_val_pred,
            'metrics': {
                'train': {'pearson': train_pearson},
                'val': {'pearson': val_pearson},
            }
        })

    # Summary
    avg_val = np.mean([r['metrics']['val']['pearson'] for r in results])
    print(f"\n✓ Avg Val Pearson across {len(seeds)} seeds: {avg_val:.6f}")
    return results


# ---------------------------------------------------------------------------
#  MLP (PyTorch)
# ---------------------------------------------------------------------------

class _MLP_Net:
    """Lightweight wrapper so the module can be imported without torch installed."""
    _net_class = None

    @classmethod
    def get_class(cls):
        if cls._net_class is None:
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self, input_dim, hidden1=256, hidden2=128, hidden3=64, dropout=0.3):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden1),
                        nn.BatchNorm1d(hidden1),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden1, hidden2),
                        nn.BatchNorm1d(hidden2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden2, hidden3),
                        nn.BatchNorm1d(hidden3),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden3, 1),
                    )

                def forward(self, x):
                    return self.net(x).squeeze(-1)

            cls._net_class = Net
        return cls._net_class


def train_mlp(X_train, y_train, X_val, y_val,
              seeds: list = None,
              hidden1: int = 256, hidden2: int = 128, hidden3: int = 64,
              dropout: float = 0.3,
              lr: float = 1e-3, weight_decay: float = 1e-5,
              epochs: int = 100, batch_size: int = 1024,
              patience: int = 10):
    """
    Train a 3-layer MLP regressor with multiple random seeds.

    Architecture: Input → 256 → BN → ReLU → Drop → 128 → BN → ReLU → Drop → 64 → BN → ReLU → Drop → 1
    Loss: 0.6*MSE + 0.4*(1-Pearson)  |  Val metric: Pearson correlation

    Args:
        X_train, y_train: Training data (numpy / pandas).
        X_val, y_val: Validation data.
        seeds: Random seeds (default: [42, 123, 456]).
        hidden1, hidden2, hidden3: Hidden layer sizes.
        dropout: Dropout probability.
        lr: Learning rate for Adam.
        weight_decay: L2 penalty for Adam.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        patience: Early-stop patience (epochs without val improvement).

    Returns:
        list[dict]: One per seed with keys
            'seed', 'state_dict', 'input_dim', 'hidden1', 'hidden2',
            'hidden3', 'dropout', 'best_epoch', 'metrics'.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if seeds is None:
        seeds = [42, 123, 456]

    # Convert to tensors
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values.astype(np.float32)
    else:
        X_train_np = np.asarray(X_train, dtype=np.float32)
    if hasattr(y_train, 'values'):
        y_train_np = y_train.values.astype(np.float32)
    else:
        y_train_np = np.asarray(y_train, dtype=np.float32)
    if hasattr(X_val, 'values'):
        X_val_np = X_val.values.astype(np.float32)
    else:
        X_val_np = np.asarray(X_val, dtype=np.float32)
    if hasattr(y_val, 'values'):
        y_val_np = y_val.values.astype(np.float32)
    else:
        y_val_np = np.asarray(y_val, dtype=np.float32)

    X_t = torch.from_numpy(X_train_np).to(device)
    y_t = torch.from_numpy(y_train_np).to(device)
    X_v = torch.from_numpy(X_val_np).to(device)
    y_v = torch.from_numpy(y_val_np).to(device)

    train_ds = TensorDataset(X_t, y_t)
    input_dim = X_t.shape[1]

    NetClass = _MLP_Net.get_class()

    print(f"\n{'='*60}")
    print("MLP MULTI-SEED TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Architecture: {input_dim} → {hidden1} → {hidden2} → {hidden3} → 1")
    print(f"Seeds: {seeds} | Epochs: {epochs} | BS: {batch_size} | LR: {lr}")
    print(f"Loss: 0.6*MSE + 0.4*(1-Pearson) | Val metric: Pearson")

    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = NetClass(input_dim, hidden1, hidden2, hidden3, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        mse_criterion = nn.MSELoss()

        def pearson_loss(pred, target):
            """Differentiable 1 - Pearson correlation."""
            vx = pred - pred.mean()
            vy = target - target.mean()
            corr = (vx * vy).sum() / (
                torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8
            )
            return 1.0 - corr

        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator(device='cpu').manual_seed(seed))

        best_val_pearson = -float('inf')
        best_state = None
        best_epoch = 0
        wait = 0

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            # --- train ---
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = 0.6 * mse_criterion(pred, yb) + 0.4 * pearson_loss(pred, yb)
                loss.backward()
                optimizer.step()

            # --- validate ---
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v).cpu().numpy()
            val_pearson = float(np.corrcoef(y_val_np, val_pred)[0, 1])

            if val_pearson > best_val_pearson:
                best_val_pearson = val_pearson
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1

            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Val Pearson: {val_pearson:.6f} "
                      f"(best: {best_val_pearson:.6f} @ ep {best_epoch})")

            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        # Final metrics with best model
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred = model(X_t).cpu().numpy()
            val_pred = model(X_v).cpu().numpy()
        train_pearson = float(np.corrcoef(y_train_np, train_pred)[0, 1])
        val_pearson_final = float(np.corrcoef(y_val_np, val_pred.ravel())[0, 1])

        elapsed = time.time() - start_time
        print(f"  Best epoch: {best_epoch} | "
              f"Train Pearson: {train_pearson:.6f} | "
              f"Val Pearson: {best_val_pearson:.6f} | "
              f"Time: {elapsed:.1f}s")

        results.append({
            'seed': seed,
            'state_dict': best_state,
            'input_dim': input_dim,
            'hidden1': hidden1,
            'hidden2': hidden2,
            'hidden3': hidden3,
            'dropout': dropout,
            'best_epoch': best_epoch,
            'y_val_pred': val_pred.ravel(),
            'metrics': {
                'train': {'pearson': train_pearson},
                'val': {'pearson': best_val_pearson},
            }
        })

    avg_val = np.mean([r['metrics']['val']['pearson'] for r in results])
    print(f"\n✓ Avg Val Pearson across {len(seeds)} seeds: {avg_val:.6f}")
    return results

