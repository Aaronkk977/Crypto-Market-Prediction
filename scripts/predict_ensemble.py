#!/usr/bin/env python
"""
Generate test predictions using ensemble model (α * XGB + (1-α) * MLP).

Trains XGBoost and MLP on the FULL training set, then predicts on test data.

Usage:
    python scripts/predict_ensemble.py
    python scripts/predict_ensemble.py --alpha 0.1
    python scripts/predict_ensemble.py --alpha 0.1 --output submission.csv
"""

import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.models import train_xgboost, train_mlp, scale_features


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_selected_features(filepath):
    """Load selected feature names from a text file (one per line, # = comment)."""
    features = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction (XGB + MLP)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Blending weight: α*XGB + (1-α)*MLP (default: 0.1)')
    parser.add_argument('--output', type=str, default='submission_ensemble.csv',
                        help='Output filename (default: submission_ensemble.csv)')
    parser.add_argument('--test-data', type=str, default='test.parquet',
                        help='Test data filename in data/ (default: test.parquet)')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    output_dir = project_dir / 'artifacts' / 'predictions'
    output_dir.mkdir(exist_ok=True, parents=True)

    alpha = args.alpha

    print("\n" + "=" * 60)
    print(f"ENSEMBLE PREDICTION (α={alpha:.1f})")
    print(f"  α*XGB + (1-α)*MLP = {alpha:.1f}*XGB + {1-alpha:.1f}*MLP")
    print("=" * 60)

    # ── Load configs ──
    xgb_cfg = load_config(project_dir / 'configs' / 'xgb.yaml')
    mlp_cfg = load_config(project_dir / 'configs' / 'mlp.yaml')

    # ── Load training data ──
    print("\nLoading training data...")
    df_train = load_data(data_dir / 'train_shap_filtered.parquet')
    target_col = 'label'

    # Features: use SHAP-selected features if configured
    feat_file = xgb_cfg.get('features', {}).get('selected_features_file')
    if feat_file:
        feat_path = project_dir / feat_file
        if feat_path.exists():
            feature_cols = [f for f in load_selected_features(feat_path) if f in df_train.columns]
            print(f"Using {len(feature_cols)} selected features from {feat_path.name}")
        else:
            meta_cols = ['label', 'symbol', 'date_id', 'time_id', 'time_group_id']
            feature_cols = [c for c in df_train.columns if c not in meta_cols]
            print(f"Feature file not found, using all {len(feature_cols)} features")
    else:
        meta_cols = ['label', 'symbol', 'date_id', 'time_id', 'time_group_id']
        feature_cols = [c for c in df_train.columns if c not in meta_cols]
    print(f"Features: {len(feature_cols)}")

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    # ── Load test data ──
    print("Loading test data...")
    df_test = load_data(data_dir / args.test_data)

    # Save IDs for submission
    if df_test.index.name == 'ID':
        test_ids = df_test.index.values
    else:
        test_ids = np.arange(1, len(df_test) + 1)

    # Drop label (dummy zeros in test) and keep same features
    available_features = [c for c in feature_cols if c in df_test.columns]
    X_test = df_test[available_features].values
    print(f"Test samples: {len(X_test)}")

    # ── Train XGBoost on full data ──
    # Use a small holdout from end of training for early stopping
    n_holdout = int(len(X_train) * 0.1)
    X_tr_xgb, X_es_xgb = X_train[:-n_holdout], X_train[-n_holdout:]
    y_tr_xgb, y_es_xgb = y_train[:-n_holdout], y_train[-n_holdout:]

    xgb_params = dict(xgb_cfg['model'])
    seeds = xgb_cfg['training']['seeds']

    print(f"\n{'='*60}")
    print("TRAINING XGBOOST ON FULL DATA")
    print(f"{'='*60}")

    xgb_results = train_xgboost(
        X_tr_xgb, y_tr_xgb, X_es_xgb, y_es_xgb,
        params=xgb_params,
        seeds=seeds,
        num_boost_round=xgb_cfg['training']['num_boost_round'],
        early_stopping_rounds=xgb_cfg['training']['early_stopping_rounds'],
    )

    # Average XGB predictions across seeds
    import xgboost as xgb
    xgb_preds = []
    for r in xgb_results:
        dtest = xgb.DMatrix(X_test)
        best_iter = r['best_iteration']
        pred = r['model'].predict(dtest, iteration_range=(0, best_iter + 1))
        xgb_preds.append(pred)
    xgb_pred = np.column_stack(xgb_preds).mean(axis=1)
    print(f"\nXGB test predictions: mean={xgb_pred.mean():.6f}, std={xgb_pred.std():.6f}")

    # ── Train MLP on full data ──
    print(f"\n{'='*60}")
    print("TRAINING MLP ON FULL DATA")
    print(f"{'='*60}")

    # Scale features
    X_tr_sc, X_es_sc, scaler = scale_features(
        pd.DataFrame(X_tr_xgb, columns=available_features),
        pd.DataFrame(X_es_xgb, columns=available_features),
    )
    X_test_sc = scaler.transform(X_test)

    model_cfg = mlp_cfg['model']
    mlp_results = train_mlp(
        X_tr_sc, y_tr_xgb, X_es_sc, y_es_xgb,
        seeds=mlp_cfg['training']['seeds'],
        hidden1=model_cfg.get('hidden1', 256),
        hidden2=model_cfg.get('hidden2', 128),
        hidden3=model_cfg.get('hidden3', 64),
        dropout=model_cfg.get('dropout', 0.3),
        lr=model_cfg.get('lr', 1e-3),
        weight_decay=model_cfg.get('weight_decay', 1e-5),
        epochs=model_cfg.get('epochs', 100),
        batch_size=model_cfg.get('batch_size', 1024),
        patience=model_cfg.get('patience', 10),
    )

    # Average MLP predictions across seeds
    import torch
    from myproj.models import _MLP_Net
    Net = _MLP_Net.get_class()

    mlp_preds = []
    for r in mlp_results:
        model = Net(r['input_dim'], r['hidden1'], r['hidden2'], r['hidden3'], r['dropout'])
        model.load_state_dict(r['state_dict'])
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test_sc)
            pred = model(X_t).numpy().ravel()
        mlp_preds.append(pred)
    mlp_pred = np.column_stack(mlp_preds).mean(axis=1)
    print(f"\nMLP test predictions: mean={mlp_pred.mean():.6f}, std={mlp_pred.std():.6f}")

    # ── Blend ──
    ensemble_pred = alpha * xgb_pred + (1 - alpha) * mlp_pred

    print(f"\n{'='*60}")
    print("ENSEMBLE PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Blend: {alpha:.1f}*XGB + {1-alpha:.1f}*MLP")
    print(f"Predictions: {len(ensemble_pred)}")
    print(f"  Mean: {ensemble_pred.mean():.6f}")
    print(f"  Std:  {ensemble_pred.std():.6f}")
    print(f"  Min:  {ensemble_pred.min():.6f}")
    print(f"  Max:  {ensemble_pred.max():.6f}")

    # ── Save submission ──
    pred_df = pd.DataFrame({
        'ID': test_ids,
        'prediction': ensemble_pred,
    })
    output_path = output_dir / args.output
    pred_df.to_csv(output_path, index=False)

    print(f"\n💾 Submission saved → {output_path}")
    print(f"   Rows: {len(pred_df)}")
    print("\n" + "=" * 60)
    print("Ensemble prediction completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
