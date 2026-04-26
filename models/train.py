"""
Model training with time-series cross-validation.
Supports LightGBM, XGBoost, CatBoost via a unified interface.

LEAKAGE PREVENTION STRATEGY:
All target-dependent features (lags, rolling means, target encodings) use
shift(FORECAST_HORIZON) or larger. Since CV folds also use FORECAST_HORIZON-day
windows, validation rows at time t only see data from t-16 or earlier, which is
always in the training period. This makes per-fold recomputation unnecessary.
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from config import TARGET, DATE_COL, SEED, CV_CONFIG, TIME_BUDGET, TRAIN_START_DATE


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def _get_cv_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-series CV: last N windows of forecast_horizon days each."""
    train_df = df[df["is_train"]].copy()
    dates = sorted(train_df[DATE_COL].unique())
    horizon = CV_CONFIG["forecast_horizon"]
    n_splits = CV_CONFIG["n_splits"]
    gap = CV_CONFIG.get("gap", 0)

    splits = []
    for i in range(n_splits):
        val_end_idx = len(dates) - i * horizon
        val_start_idx = val_end_idx - horizon
        if val_start_idx <= 0:
            break

        val_dates = set(dates[val_start_idx:val_end_idx])
        train_dates = set(dates[:max(0, val_start_idx - gap)])

        train_mask = train_df[DATE_COL].isin(train_dates).values
        val_mask = train_df[DATE_COL].isin(val_dates).values

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


def _build_model(model_name: str, params: dict):
    """Factory for model objects."""
    if model_name == "lightgbm":
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": SEED,
            "verbosity": -1,
            "n_jobs": -1,
        }
        defaults.update(params)
        return lgb.LGBMRegressor(**defaults)

    elif model_name == "xgboost":
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": SEED,
            "verbosity": 0,
            "n_jobs": -1,
        }
        defaults.update(params)
        return xgb.XGBRegressor(**defaults)

    elif model_name == "catboost":
        defaults = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 8,
            "subsample": 0.8,
            "random_seed": SEED,
            "verbose": 0,
            "thread_count": -1,
        }
        defaults.update(params)
        return cb.CatBoostRegressor(**defaults)

    raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "lightgbm",
    params: dict | None = None,
    **kwargs,
) -> dict:
    """
    Train with time-series CV and return score + timing.
    Features are already leakage-free (shift >= FORECAST_HORIZON).
    LightGBM handles NaN natively, no fillna needed.
    """
    params = params or {}
    train_df = df[df["is_train"]].copy()
    train_df = train_df[train_df[DATE_COL] >= TRAIN_START_DATE]

    # Validate features exist and are numeric
    valid_features = [
        c for c in feature_cols
        if c in train_df.columns
        and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8]
    ]

    if not valid_features:
        return {
            "score": float("inf"),
            "scores_per_fold": [],
            "elapsed": 0,
            "model_name": model_name,
            "feature_cols": [],
            "error": "No valid features",
        }

    X = train_df[valid_features].values
    y = train_df[TARGET].values
    sample_weights = _compute_sample_weights(train_df[DATE_COL])

    splits = _get_cv_splits(train_df)
    fold_scores = []

    start = time.time()

    for train_idx, val_idx in splits:
        elapsed = time.time() - start
        if elapsed > TIME_BUDGET["max_per_model_cv"]:
            break

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        model = _build_model(model_name, params)

        if model_name == "lightgbm":
            model.fit(
                X_tr, np.log1p(y_tr),
                sample_weight=w_tr,
                eval_set=[(X_val, np.log1p(y_val))],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif model_name == "xgboost":
            model.fit(
                X_tr, np.log1p(y_tr),
                sample_weight=w_tr,
                eval_set=[(X_val, np.log1p(y_val))],
                verbose=False,
            )
        elif model_name == "catboost":
            model.fit(
                X_tr, np.log1p(y_tr),
                sample_weight=w_tr,
                eval_set=(X_val, np.log1p(y_val)),
                early_stopping_rounds=50,
            )

        preds = np.expm1(model.predict(X_val))
        preds = np.clip(preds, 0, None)
        fold_scores.append(rmsle(y_val, preds))

    elapsed = time.time() - start

    return {
        "score": float(np.mean(fold_scores)) if fold_scores else float("inf"),
        "scores_per_fold": fold_scores,
        "elapsed": elapsed,
        "model_name": model_name,
        "feature_cols": valid_features,
    }


def _compute_sample_weights(dates: pd.Series, half_life_days: int = 180) -> np.ndarray:
    """Exponential decay weights: recent data gets higher weight."""
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values.astype(float)
    weights = np.exp(-0.693 * days_ago / half_life_days)
    weights = weights / weights.mean()
    return weights


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "lightgbm",
    params: dict | None = None,
):
    """Train on all training data with temporal sample weighting."""
    params = params or {}
    train_df = df[df["is_train"]].copy()
    train_df = train_df[train_df[DATE_COL] >= TRAIN_START_DATE]

    valid_features = [
        c for c in feature_cols
        if c in train_df.columns
        and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8]
    ]

    X = train_df[valid_features].values
    y = np.log1p(train_df[TARGET].values)
    weights = _compute_sample_weights(train_df[DATE_COL])

    model = _build_model(model_name, params)
    model.fit(X, y, sample_weight=weights)

    return model, valid_features
