#!/usr/bin/env python
"""
Feature Selection Experiment – compare three feature sets using walk-forward CV.

    A: All features (780+)
    B: Pearson Top-100 anonymous features
    C: SHAP-refined features (stage-2 output)

Usage:
    python scripts/run_feature_experiment.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import walk_forward_split
from myproj.models import scale_features, train_ridge_cv
from myproj.metrics import calculate_metrics


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results' / 'feature_experiment'


def run_experiment(df, feature_cols, experiment_name, target_col='label',
                   date_col='date_id', train_months=4, gap_months=1, val_months=4):
    """Run walk-forward CV on a given feature set and return per-fold metrics."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    print(f"Features: {len(feature_cols)}")

    # Build sub-df with only the needed columns (date info is in the index)
    keep = [target_col] + [c for c in feature_cols if c in df.columns]
    if date_col in df.columns:
        keep = [date_col] + keep
    df_sub = df[keep].copy()

    fold_gen = walk_forward_split(
        df_sub, date_col=date_col, target_col=target_col,
        train_months=train_months, gap_months=gap_months, val_months=val_months)

    fold_results = []
    for fold_idx, X_train, X_val, y_train, y_val in fold_gen:
        X_train_sc, X_val_sc, _ = scale_features(X_train, X_val)
        model = train_ridge_cv(X_train_sc, y_train,
                               alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                               cv=3, scoring='pearson')

        y_val_pred = model.predict(X_val_sc)
        val_metrics = calculate_metrics(y_val, y_val_pred, prefix=f"  Fold {fold_idx}")

        fold_results.append({
            'fold': fold_idx,
            'val_pearson': val_metrics['pearson'],
            'best_alpha': float(model.alpha_),
        })

    avg_pearson = np.mean([r['val_pearson'] for r in fold_results])
    std_pearson = np.std([r['val_pearson'] for r in fold_results])

    print(f"\n  → Avg Val Pearson: {avg_pearson:.6f} ± {std_pearson:.6f}")

    return {
        'experiment': experiment_name,
        'n_features': len(feature_cols),
        'avg_val_pearson': round(avg_pearson, 6),
        'std_val_pearson': round(std_pearson, 6),
        'folds': fold_results,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION EXPERIMENT  (A / B / C)")
    print("=" * 70)

    # Load full training data
    df = load_data(DATA_DIR / 'train.parquet')

    # Identify feature groups
    anon_features = sorted([c for c in df.columns if c.startswith('X')])
    all_feature_cols = [c for c in df.columns
                        if c not in ('label', 'symbol', 'date_id', 'time_id',
                                     'time_group_id', 'Y')]
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Anonymous (X*): {len(anon_features)}")
    print(f"All usable features: {len(all_feature_cols)}")

    # ── Experiment A: all features ──
    result_a = run_experiment(df, all_feature_cols, 'A: All Features')

    # ── Experiment B: Pearson Top 100 ──
    # Compute Pearson inline (quick)
    corrs = df[anon_features].corrwith(df['label']).abs().sort_values(ascending=False)
    top100 = corrs.head(100).index.tolist()
    result_b = run_experiment(df, top100, 'B: Pearson Top-100 Anonymous')

    # ── Experiment C: SHAP refined ──
    shap_feat_file = RESULTS_DIR.parent / 'feature_selection' / 'shap_selected_features.txt'
    if shap_feat_file.exists():
        shap_features = [l.strip() for l in open(shap_feat_file)
                         if l.strip() and not l.startswith('#')]
        result_c = run_experiment(df, shap_features, 'C: SHAP-Refined Features')
    else:
        print(f"\n⚠️  {shap_feat_file.name} not found — skipping Experiment C.")
        print("   Run: python scripts/feature_selection.py  first.")
        result_c = None

    # ── Comparison table ──
    results = [result_a, result_b]
    if result_c:
        results.append(result_c)

    comp_df = pd.DataFrame([{
        'Experiment': r['experiment'],
        'N Features': r['n_features'],
        'Val Pearson': r['avg_val_pearson'],
        '± std': r['std_val_pearson'],
    } for r in results])

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    comp_df.to_csv(RESULTS_DIR / 'comparison.csv', index=False)
    with open(RESULTS_DIR / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")

    # Log best to tracker
    try:
        from experiment_tracker import log_experiment
        for r in results:
            log_experiment(
                name=r['experiment'],
                params={'n_features': r['n_features']},
                val_pearson=r['avg_val_pearson'],
                notes='Feature selection experiment (Ridge baseline)'
            )
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print("Feature experiment completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
