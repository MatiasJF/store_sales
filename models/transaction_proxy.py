"""
Predict store-level transactions for test period.
Transactions are strong predictors of sales but unavailable in test.
We build a lightweight model using features available in both train and test.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from config import DATE_COL, STORE_COL, SEED, DATA_FILES


def build_transaction_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Train a transaction predictor and add predicted_transactions to df."""
    tx = pd.read_csv(DATA_FILES["transactions"], parse_dates=["date"])

    # Build features for the transaction model (all available in test)
    tx["day_of_week"] = tx["date"].dt.dayofweek
    tx["day_of_month"] = tx["date"].dt.day
    tx["month"] = tx["date"].dt.month
    tx["is_weekend"] = (tx["day_of_week"] >= 5).astype(int)

    # Add promo count per store-date from main df
    promo_counts = (
        df.groupby([STORE_COL, DATE_COL])["onpromotion"]
        .sum()
        .reset_index()
        .rename(columns={"onpromotion": "promo_count", DATE_COL: "date"})
    )
    tx = tx.merge(promo_counts, on=[STORE_COL, "date"], how="left")
    tx["promo_count"] = tx["promo_count"].fillna(0)

    # Add store-level historical avg transactions (shifted to avoid leakage)
    tx = tx.sort_values(["store_nbr", "date"])
    tx["tx_rmean_28"] = (
        tx.groupby("store_nbr")["transactions"]
        .transform(lambda x: x.shift(16).rolling(28, min_periods=1).mean())
    )

    feat_cols = ["store_nbr", "day_of_week", "day_of_month", "month",
                 "is_weekend", "promo_count", "tx_rmean_28"]

    # Train on recent data
    tx_train = tx[tx["date"] >= "2016-01-01"].dropna(subset=feat_cols)

    X = tx_train[feat_cols].values
    y = tx_train["transactions"].values

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Predict for all store-date combos in main df
    pred_df = df[[STORE_COL, DATE_COL]].drop_duplicates().copy()
    pred_df["day_of_week"] = pred_df[DATE_COL].dt.dayofweek
    pred_df["day_of_month"] = pred_df[DATE_COL].dt.day
    pred_df["month"] = pred_df[DATE_COL].dt.month
    pred_df["is_weekend"] = (pred_df["day_of_week"] >= 5).astype(int)

    # Add promo count
    pred_df = pred_df.merge(promo_counts, on=[STORE_COL, "date"], how="left")
    pred_df["promo_count"] = pred_df["promo_count"].fillna(0)

    # Add historical tx mean — merge from tx data for train dates, predict for test
    tx_hist = tx[[STORE_COL, "date", "tx_rmean_28"]].copy()
    pred_df = pred_df.merge(tx_hist, on=[STORE_COL, "date"], how="left")

    # For test dates where tx_rmean_28 is NaN, use last known value
    last_known = tx.groupby("store_nbr")["tx_rmean_28"].last().reset_index()
    last_known.columns = ["store_nbr", "tx_rmean_28_fallback"]
    pred_df = pred_df.merge(last_known, on="store_nbr", how="left")
    pred_df["tx_rmean_28"] = pred_df["tx_rmean_28"].fillna(pred_df["tx_rmean_28_fallback"])
    pred_df.drop(columns=["tx_rmean_28_fallback"], inplace=True)

    X_pred = pred_df[feat_cols].values
    pred_df["predicted_transactions"] = model.predict(X_pred).clip(0)

    # Merge back to main df
    df = df.merge(
        pred_df[[STORE_COL, DATE_COL, "predicted_transactions"]],
        on=[STORE_COL, DATE_COL],
        how="left",
    )

    return df
