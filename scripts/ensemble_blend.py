#!/usr/bin/env python
"""
Ensemble Blending – loads saved val predictions from XGBoost and MLP,
then grid-searches the optimal blending weight α.

    final_pred = α * XGB + (1 - α) * MLP

Prerequisites:
    python scripts/train_xgb.py   # saves results/xgb_cv/val_preds/
    python scripts/train_mlp.py   # saves results/mlp_cv/val_preds/

Usage:
    python scripts/ensemble_blend.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

PROJECT_DIR = Path(__file__).parent.parent
XGB_PRED_DIR = PROJECT_DIR / 'results' / 'xgb_cv' / 'val_preds'
MLP_PRED_DIR = PROJECT_DIR / 'results' / 'mlp_cv' / 'val_preds'


def discover_folds():
    """Find fold numbers that have both XGB and MLP predictions."""
    xgb_files = sorted(XGB_PRED_DIR.glob('fold*_xgb_pred.npy'))
    folds = []
    for f in xgb_files:
        fold_idx = int(f.stem.split('_')[0].replace('fold', ''))
        mlp_file = MLP_PRED_DIR / f'fold{fold_idx}_mlp_pred.npy'
        y_true_file = XGB_PRED_DIR / f'fold{fold_idx}_y_true.npy'
        if mlp_file.exists() and y_true_file.exists():
            folds.append(fold_idx)
    return sorted(folds)


def load_fold(fold_idx):
    """Load XGB preds, MLP preds, and y_true for a fold."""
    xgb_pred = np.load(XGB_PRED_DIR / f'fold{fold_idx}_xgb_pred.npy')
    mlp_pred = np.load(MLP_PRED_DIR / f'fold{fold_idx}_mlp_pred.npy')
    y_true = np.load(XGB_PRED_DIR / f'fold{fold_idx}_y_true.npy')
    return xgb_pred, mlp_pred, y_true


def pearson(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def main():
    print("\n" + "=" * 60)
    print("ENSEMBLE BLENDING (from saved predictions)")
    print("=" * 60)

    # Check prerequisites
    if not XGB_PRED_DIR.exists():
        print(f"❌ XGB predictions not found: {XGB_PRED_DIR}")
        print("   Run: python scripts/train_xgb.py")
        return
    if not MLP_PRED_DIR.exists():
        print(f"❌ MLP predictions not found: {MLP_PRED_DIR}")
        print("   Run: python scripts/train_mlp.py")
        return

    folds = discover_folds()
    if not folds:
        print("❌ No matching folds found. Ensure both XGB and MLP have been trained.")
        return

    print(f"Found {len(folds)} folds: {folds}")

    # --- Individual model performance ---
    print(f"\n{'─'*60}")
    print("Individual Model Performance (per fold)")
    print(f"{'─'*60}")

    for fold_idx in folds:
        xgb_pred, mlp_pred, y_true = load_fold(fold_idx)
        print(f"  Fold {fold_idx}: XGB Pearson={pearson(y_true, xgb_pred):.6f}, "
              f"MLP Pearson={pearson(y_true, mlp_pred):.6f}")

    # --- Grid search α ---
    alphas = np.arange(0.0, 1.05, 0.1)  # 0.0, 0.1, ..., 1.0

    print(f"\n{'─'*60}")
    print("Grid Search: α * XGB + (1-α) * MLP")
    print(f"{'─'*60}")

    alpha_results = []
    for alpha in alphas:
        fold_pearsons = []
        for fold_idx in folds:
            xgb_pred, mlp_pred, y_true = load_fold(fold_idx)
            blended = alpha * xgb_pred + (1 - alpha) * mlp_pred
            fold_pearsons.append(pearson(y_true, blended))

        avg = np.mean(fold_pearsons)
        std = np.std(fold_pearsons)
        alpha_results.append({
            'alpha': round(alpha, 1),
            'avg_pearson': avg,
            'std_pearson': std,
            'per_fold': fold_pearsons,
        })
        print(f"  α={alpha:.1f}  →  Avg Pearson={avg:.6f} ± {std:.6f}")

    # --- Best α ---
    best = max(alpha_results, key=lambda x: x['avg_pearson'])
    print(f"\n{'='*60}")
    print(f"BEST: α={best['alpha']:.1f}  →  Avg Val Pearson={best['avg_pearson']:.6f}")
    print(f"  (α=1.0 means pure XGB, α=0.0 means pure MLP)")
    print(f"{'='*60}")

    # --- Save results ---
    results_dir = PROJECT_DIR / 'results' / 'ensemble'
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        'alpha': r['alpha'],
        'avg_pearson': round(r['avg_pearson'], 6),
        'std_pearson': round(r['std_pearson'], 6),
    } for r in alpha_results])
    df.to_csv(results_dir / 'blend_search.csv', index=False)
    print(f"\nResults saved to {results_dir / 'blend_search.csv'}")

    # --- Log to experiment tracker ---
    try:
        from experiment_tracker import log_experiment
        log_experiment(
            name=f'Ensemble (α={best["alpha"]:.1f})',
            params={'alpha': best['alpha'], 'method': 'xgb_mlp_blend'},
            val_pearson=best['avg_pearson'],
            notes=f'α*XGB + (1-α)*MLP, best α={best["alpha"]:.1f}, '
                  f'{len(folds)}-fold walk-forward'
        )
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("Ensemble blending completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
