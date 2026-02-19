#!/usr/bin/env python
"""
Three-Stage Feature Selection Pipeline
========================================
Stage 1: Pearson correlation filter  →  Remove near-zero-correlation features (X*)
Stage 2: SHAP refinement via walk-forward XGBoost  →  final feature set
Stage 3: AutoEncoder  →  synthesise deep bottleneck features appended to output

Usage:
    python scripts/feature_selection.py
    python scripts/feature_selection.py --corr_threshold 1e-4 --shap_top 50
    python scripts/feature_selection.py --ae_dim 8 --ae_epochs 50
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myproj.data import load_data
from myproj.split import walk_forward_split


# ──────────────────────────────────────────────────────────────────────
#  Stage 1  –  Pearson correlation filter
# ──────────────────────────────────────────────────────────────────────

def stage1_pearson_filter(df: pd.DataFrame,
                          target_col: str = 'label',
                          corr_threshold: float = 1e-4) -> list:
    """
    Compute |Pearson(Xi, label)| for every anonymous feature (X*) and
    return features whose |correlation| > corr_threshold, sorted by
    descending |correlation|.  Features with |corr| <= threshold are
    considered absolutely uncorrelated and are removed.
    """
    print(f"\n{'='*60}")
    print("STAGE 1 – PEARSON CORRELATION FILTER")
    print(f"{'='*60}")

    # Only consider anonymous features (prefix 'X')
    anon_features = [c for c in df.columns if c.startswith('X')]
    print(f"Anonymous features found: {len(anon_features)}")

    y = df[target_col]
    corr_records = []
    for feat in anon_features:
        rho = df[feat].corr(y)
        corr_records.append({'feature': feat, 'pearson': rho, 'abs_pearson': abs(rho)})

    corr_df = pd.DataFrame(corr_records).sort_values('abs_pearson', ascending=False)
    corr_df.reset_index(drop=True, inplace=True)

    # Keep only features above the correlation threshold
    mask = corr_df['abs_pearson'] > corr_threshold
    corr_df['kept'] = mask
    selected = corr_df.loc[mask, 'feature'].tolist()
    removed = corr_df.loc[~mask, 'feature'].tolist()

    print(f"Correlation threshold: |corr| > {corr_threshold:.1e}")
    print(f"Features kept:    {len(selected)}")
    print(f"Features removed: {len(removed)}")
    if len(selected) > 0:
        print(f"  |correlation| range: "
              f"{corr_df.loc[mask, 'abs_pearson'].iloc[0]:.6f} → "
              f"{corr_df.loc[mask, 'abs_pearson'].iloc[-1]:.6f}")
    if len(removed) > 0:
        print(f"  Removed features: {removed}")

    return selected, corr_df


# ──────────────────────────────────────────────────────────────────────
#  Stage 2  –  SHAP refinement via walk-forward CV
# ──────────────────────────────────────────────────────────────────────

def stage2_shap_refinement(df: pd.DataFrame,
                           stage1_features: list,
                           target_col: str = 'label',
                           shap_top: int = 50,
                           train_months: int = 4,
                           gap_months: int = 1,
                           val_months: int = 4,
                           date_col: str = 'date_id',
                           xgb_params: dict | None = None,
                           results_dir: Path | None = None) -> list:
    """
    Walk-forward SHAP feature selection.

    For each fold produced by walk_forward_split:
      1. Train a lightweight XGBoost on the training window.
      2. Compute SHAP values on the **validation** window.
      3. Rank features by mean |SHAP| and select top-K.
    The final feature set is the **union** of per-fold top-K lists.

    Returns:
        selected: list of selected feature names (union of per-fold top-K)
        agg_shap_df: DataFrame with per-feature aggregated SHAP statistics
    """
    import xgboost as xgb
    import shap

    print(f"\n{'='*60}")
    print("STAGE 2 – SHAP REFINEMENT (WALK-FORWARD CV)")
    print(f"{'='*60}")
    print(f"Top-K per fold: {shap_top}")

    default_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'verbosity': 0,
        'random_state': 42,
        'n_estimators': 300,
    }
    if xgb_params:
        default_params.update(xgb_params)

    # Prepare a sub-DataFrame with stage-1 features + metadata needed for splitting
    keep_cols = list(stage1_features) + [target_col]
    if date_col in df.columns and date_col not in keep_cols:
        keep_cols.append(date_col)
    df_sub = df[keep_cols].copy()

    # Walk-forward CV generator
    folds = walk_forward_split(
        df_sub, date_col=date_col, target_col=target_col,
        drop_cols=None, train_months=train_months,
        gap_months=gap_months, val_months=val_months,
    )

    # Per-fold SHAP accumulation
    union_features: set = set()
    fold_shap_records: list[dict] = []  # {feature: mean_abs_shap} per fold
    n_folds = 0

    for fold_idx, X_train, X_val, y_train, y_val in folds:
        n_folds += 1
        print(f"\n── Fold {fold_idx} ─────────────────────────────")

        # Drop date_col if it leaked into features
        for col in [date_col]:
            if col in X_train.columns:
                X_train = X_train.drop(columns=[col])
                X_val = X_val.drop(columns=[col])

        # Keep only stage-1 features (in case of column order mismatch)
        X_train = X_train[stage1_features].fillna(0)
        X_val = X_val[stage1_features].fillna(0)

        print(f"  Training XGBoost ({len(stage1_features)} features, "
              f"{len(X_train)} train / {len(X_val)} val samples)...")
        model = xgb.XGBRegressor(**default_params)
        model.fit(X_train, y_train, verbose=False)

        # Compute SHAP on validation set (out-of-sample)
        print(f"  Computing SHAP on validation set...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        except (ValueError, Exception) as e:
            print(f"  ⚠️  SHAP failed ({e.__class__.__name__}), "
                  f"falling back to gain-based importance.")
            mean_abs_shap = model.feature_importances_

        # Build per-fold ranking
        fold_df = pd.DataFrame({
            'feature': stage1_features,
            'mean_abs_shap': mean_abs_shap,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        fold_topk = fold_df.head(shap_top)['feature'].tolist()
        union_features.update(fold_topk)

        print(f"  Top-{shap_top} features this fold:")
        print(fold_df.head(shap_top).to_string(index=False))

        # Save per-fold SHAP ranking to CSV
        if results_dir is not None:
            fold_path = results_dir / f'shap_fold_{fold_idx}.csv'
            fold_df.to_csv(fold_path, index=False)
            print(f"  📄 Saved fold {fold_idx} SHAP → {fold_path.name}")

        # Save per-fold SHAP for aggregation
        fold_shap_records.append(
            dict(zip(fold_df['feature'], fold_df['mean_abs_shap']))
        )

    # ── Aggregate SHAP across folds ──
    agg_data = {feat: [] for feat in stage1_features}
    for rec in fold_shap_records:
        for feat in stage1_features:
            agg_data[feat].append(rec.get(feat, 0.0))

    # Count how many folds each feature appeared in the top-K
    topk_counts = {f: 0 for f in stage1_features}
    for rec in fold_shap_records:
        sorted_feats = sorted(rec, key=rec.get, reverse=True)[:shap_top]
        for f in sorted_feats:
            topk_counts[f] += 1

    agg_shap_df = pd.DataFrame({
        'feature': stage1_features,
        'mean_abs_shap_avg': [np.mean(agg_data[f]) for f in stage1_features],
        'mean_abs_shap_std': [np.std(agg_data[f]) for f in stage1_features],
        'n_folds_in_topk': [topk_counts[f] for f in stage1_features],
    }).sort_values('mean_abs_shap_avg', ascending=False).reset_index(drop=True)

    # Sort union features by aggregated SHAP
    rank_map = dict(zip(agg_shap_df['feature'], agg_shap_df['mean_abs_shap_avg']))
    selected = sorted(union_features, key=lambda f: rank_map.get(f, 0), reverse=True)

    print(f"\n{'='*60}")
    print(f"STAGE 2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Folds:                {n_folds}")
    print(f"  Top-K per fold:       {shap_top}")
    print(f"  Union feature count:  {len(selected)}")
    print(f"\nTop 10 features by avg |SHAP|:")
    print(agg_shap_df.head(10).to_string(index=False))

    return selected, agg_shap_df


# ──────────────────────────────────────────────────────────────────────
#  Stage 3  –  AutoEncoder deep feature synthesis
# ──────────────────────────────────────────────────────────────────────

def stage3_autoencoder(df: pd.DataFrame,
                      features: list,
                      ae_dim: int = 8,
                      ae_epochs: int = 50,
                      ae_lr: float = 1e-3,
                      batch_size: int = 1024) -> pd.DataFrame:
    """
    Train an AutoEncoder on the selected features and return a DataFrame
    with `ae_dim` new deep features (AE_0 … AE_{ae_dim-1}) appended.

    Architecture:
        input_dim → hidden (2×ae_dim) → bottleneck (ae_dim) → hidden → input_dim

    The encoder is used to project the selected features into the
    bottleneck space, producing `ae_dim` synthetic features.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*60}")
    print("STAGE 3 – AUTOENCODER DEEP FEATURE SYNTHESIS")
    print(f"{'='*60}")

    input_dim = len(features)
    hidden_dim = max(ae_dim * 2, 32)
    print(f"  Input dim:      {input_dim}")
    print(f"  Hidden dim:     {hidden_dim}")
    print(f"  Bottleneck dim: {ae_dim}")
    print(f"  Epochs:         {ae_epochs}")
    print(f"  Learning rate:  {ae_lr}")

    # ── Prepare data ──
    X = df[features].fillna(0).values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ── Build AutoEncoder ──
    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, ae_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(ae_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat

        def encode(self, x):
            return self.encoder(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X_scaled))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Train ──
    print(f"\n  Training on {len(X_scaled)} samples (device={device})...")
    model.train()
    for epoch in range(1, ae_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat = model(batch_x)
            loss = criterion(x_hat, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        if epoch == 1 or epoch % 10 == 0 or epoch == ae_epochs:
            print(f"    Epoch {epoch:3d}/{ae_epochs}  loss={avg_loss:.6f}")

    # ── Encode all data ──
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        Z = model.encode(X_tensor).cpu().numpy()

    ae_cols = [f'AE_{i}' for i in range(ae_dim)]
    ae_df = pd.DataFrame(Z, index=df.index, columns=ae_cols)

    print(f"\n  Generated {ae_dim} deep features: {ae_cols}")
    print(f"  Bottleneck stats:")
    for col in ae_cols:
        print(f"    {col}: mean={ae_df[col].mean():.4f}  std={ae_df[col].std():.4f}")

    return ae_df


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Three-Stage Feature Selection (Pearson + SHAP + AutoEncoder)')
    parser.add_argument('--input', type=str, default='train.parquet',
                        help='Input data file in data/ (default: train.parquet)')
    parser.add_argument('--output', type=str, default='train_shap_filtered.parquet',
                        help='Output filtered parquet (default: train_shap_filtered.parquet)')
    parser.add_argument('--corr_threshold', type=float, default=1e-4,
                        help='Stage-1: remove features with |corr| <= threshold (default: 1e-4)')
    parser.add_argument('--shap_top', type=int, default=50,
                        help='Stage-2: top K per fold for union (default: 50)')
    parser.add_argument('--target_col', type=str, default='label',
                        help='Target column name (default: label)')
    parser.add_argument('--train_months', type=int, default=4,
                        help='Walk-forward: training window in months (default: 4)')
    parser.add_argument('--gap_months', type=int, default=1,
                        help='Walk-forward: gap in months (default: 1)')
    parser.add_argument('--val_months', type=int, default=4,
                        help='Walk-forward: validation window in months (default: 4)')
    parser.add_argument('--date_col', type=str, default='date_id',
                        help='Date column for walk-forward splits (default: date_id)')
    parser.add_argument('--ae_dim', type=int, default=8,
                        help='Stage-3: AutoEncoder bottleneck dimension (default: 8)')
    parser.add_argument('--ae_epochs', type=int, default=50,
                        help='Stage-3: AutoEncoder training epochs (default: 50)')
    parser.add_argument('--ae_lr', type=float, default=1e-3,
                        help='Stage-3: AutoEncoder learning rate (default: 1e-3)')

    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    results_dir = project_dir / 'results' / 'feature_selection'
    results_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 70)
    print("THREE-STAGE FEATURE SELECTION PIPELINE")
    print("=" * 70)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Stage-1 corr threshold: {args.corr_threshold:.1e}")
    print(f"Stage-2 top-K per fold: {args.shap_top}")
    print(f"Stage-3 AE bottleneck:  {args.ae_dim}")
    print(f"Walk-forward: {args.train_months}mo train | {args.gap_months}mo gap | {args.val_months}mo val")
    print("=" * 70)

    # Load data
    df = load_data(data_dir / args.input)

    # ── Stage 1 ──
    stage1_features, corr_df = stage1_pearson_filter(
        df, target_col=args.target_col, corr_threshold=args.corr_threshold)

    # Save stage-1 correlation table
    corr_path = results_dir / 'pearson_correlations.csv'
    corr_df.to_csv(corr_path, index=False)
    print(f"\n📄 Saved Pearson table → {corr_path.name}")

    # ── Stage 2 ──
    final_features, shap_df = stage2_shap_refinement(
        df, stage1_features, target_col=args.target_col,
        shap_top=args.shap_top,
        train_months=args.train_months, gap_months=args.gap_months,
        val_months=args.val_months, date_col=args.date_col,
        results_dir=results_dir)

    # Save SHAP table
    shap_path = results_dir / 'shap_feature_importance.csv'
    shap_df.to_csv(shap_path, index=False)
    print(f"📄 Saved SHAP table   → {shap_path.name}")

    # ── Stage 3 ──
    ae_df = stage3_autoencoder(
        df, final_features,
        ae_dim=args.ae_dim, ae_epochs=args.ae_epochs, ae_lr=args.ae_lr)

    # Save filtered dataset (original selected features + AE deep features)
    metadata_cols = ['symbol', 'date_id', 'time_id', 'time_group_id']
    label_cols = [args.target_col]
    keep = [c for c in metadata_cols if c in df.columns] + final_features
    keep += [c for c in label_cols if c in df.columns and c not in keep]
    df_out = df[keep].copy()

    # Append AE features
    for col in ae_df.columns:
        df_out[col] = ae_df[col].values

    out_path = data_dir / args.output
    df_out.to_parquet(out_path, index=True)
    print(f"\n💾 Saved filtered dataset → {out_path.name}  ({df_out.shape})")

    # Update feature list to include AE features
    all_features = final_features + list(ae_df.columns)
    feat_path = results_dir / 'shap_selected_features.txt'
    with open(feat_path, 'w') as f:
        f.write(f"# Three-stage feature selection: Pearson |corr|>{args.corr_threshold:.1e} "
                f"→ SHAP top-{args.shap_top}/fold union "
                f"→ +{args.ae_dim} AE deep features\n")
        for feat in all_features:
            f.write(feat + '\n')
    print(f"📄 Updated feature list → {feat_path.name}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Stage 1: {len(stage1_features)} features (Pearson |corr| > {args.corr_threshold:.1e})")
    print(f"  Stage 2: {len(final_features)} features (SHAP top-{args.shap_top}/fold union)")
    print(f"  Stage 3: +{args.ae_dim} AE deep features")
    print(f"  Total:   {len(all_features)} features")
    print(f"  Output:  {out_path}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
