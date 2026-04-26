"""
Feature importance scoring using a quick baseline model.
Runs after signal detection to confirm which features actually help prediction.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from config import TARGET, SEED


def score_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int = 100,
) -> dict[str, float]:
    """
    Train a quick LightGBM and return feature importances (gain-based).
    Only uses train data.
    """
    train = df[df["is_train"]].copy()

    # Drop rows with NaN target
    train = train.dropna(subset=[TARGET])

    # Keep only numeric features that exist and aren't all NaN
    valid_features = []
    for col in feature_cols:
        if col in train.columns and train[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            if train[col].notna().sum() > 100:
                valid_features.append(col)

    if not valid_features:
        return {}

    X = train[valid_features].fillna(-999)
    y = np.log1p(train[TARGET].values)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X, y)

    importances = dict(zip(valid_features, model.feature_importances_))
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
