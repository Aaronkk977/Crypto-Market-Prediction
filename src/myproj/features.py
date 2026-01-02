"""
Feature engineering utilities.
"""

import pandas as pd
import numpy as np

EPS = 1e-9

def add_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add order book depth and trade imbalance features.
    Only uses features available in each row (no temporal ordering dependency).
    
    Args:
        df: Input dataframe with market data
        
    Returns:
        pd.DataFrame: Dataframe with additional engineered features
    """
    print("Adding depth and imbalance features...")
    df = df.copy()
    
    # Core features
    df["imbalance_best"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + EPS)
    df["trade_imbalance"] = (df["buy_qty"] - df["sell_qty"]) / (df["buy_qty"] + df["sell_qty"] + EPS)
    df["vol_log1p"] = np.log1p(df["volume"])
    df["bid_qty_log1p"] = np.log1p(df["bid_qty"])
    df["ask_qty_log1p"] = np.log1p(df["ask_qty"])
    
    # Additional features
    total_best_qty = df["bid_qty"] + df["ask_qty"]  # Intermediate calculation
    df["total_best_qty_log1p"] = np.log1p(total_best_qty)
    
    book_to_trade_ratio = total_best_qty / (df["volume"] + EPS)  # Intermediate calculation
    df["book_to_trade_ratio_log1p"] = np.log1p(book_to_trade_ratio)
    
    # Interaction features
    df["bid_to_buy"] = np.log1p(df["bid_qty"]) * (df["buy_qty"] / (df["volume"] + EPS))
    df["ask_to_sell"] = np.log1p(df["ask_qty"]) * (df["sell_qty"] / (df["volume"] + EPS))
    
    # Select only the desired features
    feature_cols = [
        "imbalance_best", "trade_imbalance", "vol_log1p",
        "bid_qty_log1p", "ask_qty_log1p",
        "total_best_qty_log1p",
        "book_to_trade_ratio_log1p",
        "bid_to_buy", "ask_to_sell"
    ]
    
    # Convert to float32 to reduce memory usage
    for col in feature_cols:
        df[col] = df[col].astype("float32")
    
    # Keep only original columns + new features
    original_cols = [c for c in df.columns if c not in feature_cols]
    df = df[original_cols + feature_cols]
    
    print(f"Added {len(feature_cols)} new features")
    return df

def get_engineered_feature_names() -> list:
    """
    Get list of engineered feature names.
    
    Returns:
        list: Feature names
    """
    return [
        "imbalance_best", "trade_imbalance", "vol_log1p",
        "bid_qty_log1p", "ask_qty_log1p",
        "total_best_qty_log1p",
        "book_to_trade_ratio_log1p",
        "bid_to_buy", "ask_to_sell"
    ]
