"""
Two-stage zero/nonzero classifier.
Stage 1: Predict P(sales=0) per row using a LightGBM classifier.
Stage 2: Use P(zero) as a feature for the regressor + as post-processing multiplier.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from config import TARGET, DATE_COL, SEED, TRAIN_START_DATE


def train_zero_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple:
    """Train a binary classifier to predict zero vs nonzero sales.
    Returns (model, feature_cols_used).
    """
    train_df = df[df["is_train"]].copy()
    train_df = train_df[train_df[DATE_COL] >= TRAIN_START_DATE]

    valid_features = [
        c for c in feature_cols
        if c in train_df.columns
        and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8]
    ]

    X = train_df[valid_features].values
    y = (train_df[TARGET].values == 0).astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
        is_unbalance=True,
    )
    model.fit(X, y)

    return model, valid_features


def add_zero_probability(
    df: pd.DataFrame,
    zero_model,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Add P(zero) column to the dataframe for all rows."""
    X = df[feature_cols].values
    proba = zero_model.predict_proba(X)[:, 1]  # P(sales=0)
    df["p_zero"] = proba
    return df
