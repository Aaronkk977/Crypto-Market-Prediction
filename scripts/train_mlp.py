#!/usr/bin/env python
"""
Train 3-Layer MLP with Walk-Forward CV and multi-seed ensemble.

Usage:
    python scripts/train_mlp.py
    python scripts/train_mlp.py --config configs/mlp.yaml
"""

import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
import pandas as pd
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import split_data
from myproj.models import train_mlp, scale_features


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_selected_features(filepath):
    """Load selected feature names from a text file."""
    features = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def main():
    parser = argparse.ArgumentParser(description='Train MLP (multi-seed, walk-forward CV)')
    parser.add_argument('--config', type=str, default='configs/mlp.yaml')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    config = load_config(project_dir / args.config)

    print("\n" + "=" * 60)
    print("MLP TRAINING PIPELINE")
    print("=" * 60)

    # --- Load data ---
    df = load_data(project_dir / config['data']['train_path'])

    # --- Optional feature filter ---
    feat_file = config.get('features', {}).get('selected_features_file')
    if feat_file:
        feat_path = project_dir / feat_file
        if feat_path.exists():
            selected_features = load_selected_features(feat_path)
            meta_cols = ['symbol', 'date_id', 'time_id', 'time_group_id',
                         config['data']['target_col']]
            keep = [c for c in meta_cols if c in df.columns] + \
                   [f for f in selected_features if f in df.columns]
            df = df[keep]
            print(f"Using {len(selected_features)} selected features from {feat_path.name}")

    # --- Walk-forward CV ---
    split_cfg = config['split']
    fold_gen = split_data(
        df,
        split_method=split_cfg['cv_method'],
        target_col=config['data']['target_col'],
        date_col=split_cfg.get('date_col', 'date_id'),
        train_months=split_cfg.get('train_months', 4),
        gap_months=split_cfg.get('gap_months', 1),
        val_months=split_cfg.get('val_months', 4),
        drop_cols=['symbol', 'date_id', 'time_id', 'time_group_id'],
    )

    seeds = config['training']['seeds']
    model_cfg = config['model']
    do_scale = config['training'].get('scale_features', True)
    all_fold_results = []

    for fold_idx, X_train, X_val, y_train, y_val in fold_gen:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*60}")

        # Scale features (important for MLP)
        if do_scale:
            X_train_sc, X_val_sc, scaler = scale_features(X_train, X_val)
        else:
            X_train_sc, X_val_sc = X_train, X_val

        seed_results = train_mlp(
            X_train_sc, y_train, X_val_sc, y_val,
            seeds=seeds,
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

        avg_pearson = np.mean([r['metrics']['val']['pearson'] for r in seed_results])

        # Save averaged val predictions for blending
        avg_val_preds = np.column_stack(
            [r['y_val_pred'] for r in seed_results]
        ).mean(axis=1)

        preds_dir = project_dir / config['output']['results_dir'] / 'val_preds'
        preds_dir.mkdir(parents=True, exist_ok=True)
        np.save(preds_dir / f'fold{fold_idx}_mlp_pred.npy', avg_val_preds)
        np.save(preds_dir / f'fold{fold_idx}_y_true.npy',
                y_val.values if hasattr(y_val, 'values') else np.asarray(y_val))

        all_fold_results.append({
            'fold': fold_idx,
            'avg_val_pearson': float(avg_pearson),
            'per_seed': [{
                'seed': r['seed'],
                'val_pearson': r['metrics']['val']['pearson'],
                'best_epoch': r['best_epoch'],
            } for r in seed_results]
        })

    # --- Summary ---
    print("\n" + "=" * 60)
    print("WALK-FORWARD CV SUMMARY (MLP)")
    print("=" * 60)
    for fr in all_fold_results:
        print(f"  Fold {fr['fold']}: Val Pearson={fr['avg_val_pearson']:.6f}")

    overall_pearson = np.mean([r['avg_val_pearson'] for r in all_fold_results])
    print(f"\n  Overall: Val Pearson={overall_pearson:.6f} ± "
          f"{np.std([r['avg_val_pearson'] for r in all_fold_results]):.6f}")

    # --- Save results ---
    results_dir = project_dir / config['output']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'cv_results.json', 'w') as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"\nResults saved to {results_dir / 'cv_results.json'}")

    # --- Log to experiment tracker ---
    try:
        sys.path.insert(0, str(project_dir / 'scripts'))
        from experiment_tracker import log_experiment
        log_experiment(
            name='MLP (walk-forward)',
            params=model_cfg,
            val_pearson=overall_pearson,
            notes=f"{len(all_fold_results)}-fold walk-forward, {len(seeds)} seeds"
        )
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("MLP training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
