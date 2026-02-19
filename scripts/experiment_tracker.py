#!/usr/bin/env python
"""
Experiment Tracker – log, query, and visualise experiment results.

Functions:
    log_experiment(name, params, val_pearson, notes)
    print_leaderboard()
    plot_scatter(y_true, y_pred, title, save_path)

CLI usage:
    python scripts/experiment_tracker.py --leaderboard
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_DIR / 'results' / 'final_comparison.csv'


def log_experiment(name: str,
                   params: dict,
                   val_pearson: float,
                   notes: str = ''):
    """Append one row to results/final_comparison.csv."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'experiment': name,
        'val_pearson': round(val_pearson, 6),
        'params': json.dumps(params, default=str),
        'notes': notes,
    }

    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"📝 Logged experiment '{name}' (Val Pearson={val_pearson:.6f}) → {RESULTS_FILE.name}")


def print_leaderboard():
    """Print all logged experiments sorted by Val Pearson (descending)."""
    if not RESULTS_FILE.exists():
        print("No experiments logged yet.")
        return

    df = pd.read_csv(RESULTS_FILE)
    df = df.sort_values('val_pearson', ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed rank

    print("\n" + "=" * 80)
    print("EXPERIMENT LEADERBOARD  (sorted by Val Pearson ↓)")
    print("=" * 80)
    cols = ['experiment', 'val_pearson', 'notes', 'timestamp']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string())
    print("=" * 80)


def plot_scatter(y_true, y_pred, title='Predictions vs Actuals', save_path=None):
    """Generate scatter plot of predictions vs actuals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.15, s=4, edgecolors='none')

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='Perfect')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{title}\nPearson = {pearson:.6f}')
    ax.legend()
    fig.tight_layout()

    if save_path is None:
        save_path = PROJECT_DIR / 'results' / f'scatter_{title.replace(" ", "_").lower()}.png'
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📊 Scatter plot saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Experiment Tracker')
    parser.add_argument('--leaderboard', action='store_true',
                        help='Print leaderboard of all experiments')
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
