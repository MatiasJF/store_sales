"""
Automated signal detection for machine consumption.
Outputs structured results that feed into feature selection.
"""

import numpy as np
import pandas as pd
from config import TARGET, DATE_COL, STORE_COL, FAMILY_COL


def _correlation_signals(df: pd.DataFrame) -> dict[str, float]:
    """Pearson correlation of numeric columns with target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET and c != "id"]
    corrs = {}
    for col in numeric_cols:
        valid = df[[col, TARGET]].dropna()
        if len(valid) > 100:
            corrs[col] = valid[col].corr(valid[TARGET])
    return dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))


def _seasonality_signals(df: pd.DataFrame) -> dict:
    """Detect strength of weekly, monthly, yearly patterns."""
    train = df[df["is_train"]].copy()

    # Weekly pattern: variance of mean sales by day-of-week
    if "day_of_week" in train.columns:
        dow_means = train.groupby("day_of_week")[TARGET].mean()
        weekly_strength = dow_means.std() / dow_means.mean() if dow_means.mean() > 0 else 0
    else:
        weekly_strength = None

    # Monthly pattern
    if "month" in train.columns:
        month_means = train.groupby("month")[TARGET].mean()
        monthly_strength = month_means.std() / month_means.mean() if month_means.mean() > 0 else 0
    else:
        monthly_strength = None

    # Payday pattern
    if "day_of_month" in train.columns:
        dom_means = train.groupby("day_of_month")[TARGET].mean()
        payday_ratio = (
            dom_means.loc[[15, dom_means.index.max()]].mean() / dom_means.mean()
            if dom_means.mean() > 0 else None
        )
    else:
        payday_ratio = None

    return {
        "weekly_cv": weekly_strength,
        "monthly_cv": monthly_strength,
        "payday_ratio": payday_ratio,
    }


def _zero_rate_by_family(df: pd.DataFrame) -> dict[str, float]:
    """Percentage of zero-sales rows per family."""
    train = df[df["is_train"]]
    rates = train.groupby(FAMILY_COL)[TARGET].apply(lambda x: (x == 0).mean())
    return rates.sort_values(ascending=False).to_dict()


def _oil_lag_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Test oil price at various lags against aggregated daily sales."""
    train = df[df["is_train"]].copy()
    daily = train.groupby(DATE_COL).agg({TARGET: "sum", "dcoilwtico": "first"}).dropna()

    results = {}
    for lag in [0, 7, 14, 21, 30]:
        shifted = daily["dcoilwtico"].shift(lag)
        valid = pd.DataFrame({"sales": daily[TARGET], "oil": shifted}).dropna()
        if len(valid) > 50:
            results[f"oil_lag_{lag}"] = valid["sales"].corr(valid["oil"])
    return results


def detect_signals(df: pd.DataFrame) -> dict:
    """
    Run all signal detectors. Returns a structured dict of findings.
    This is consumed by the optimizer to prioritize feature search.
    """
    signals = {}

    signals["correlations"] = _correlation_signals(df)
    signals["seasonality"] = _seasonality_signals(df)
    signals["zero_rates"] = _zero_rate_by_family(df)
    signals["oil_lags"] = _oil_lag_correlations(df)

    # Rank feature groups by signal strength
    group_scores = {}
    corrs = signals["correlations"]

    # Temporal signal strength
    temporal_cols = [c for c in corrs if c in [
        "day_of_week", "day_of_month", "month", "is_weekend",
        "days_to_payday", "is_payday"
    ]]
    if temporal_cols:
        group_scores["temporal"] = max(abs(corrs[c]) for c in temporal_cols)

    # Promotion signal strength
    promo_cols = [c for c in corrs if "promo" in c.lower()]
    if promo_cols:
        group_scores["promotions"] = max(abs(corrs[c]) for c in promo_cols)

    # Oil signal strength
    oil_cols = [c for c in corrs if "oil" in c.lower()]
    if oil_cols:
        group_scores["oil"] = max(abs(corrs[c]) for c in oil_cols)

    # Holiday signal strength
    holiday_cols = [c for c in corrs if "holiday" in c.lower()]
    if holiday_cols:
        group_scores["holidays"] = max(abs(corrs[c]) for c in holiday_cols)

    # Store signal strength
    store_cols = [c for c in corrs if c in [
        "store_nbr", "cluster", "type_encoded", "city_encoded", "state_encoded"
    ]]
    if store_cols:
        group_scores["store"] = max(abs(corrs[c]) for c in store_cols)

    # Lag signal strength
    lag_cols = [c for c in corrs if "lag" in c.lower() or "rmean" in c.lower() or "rstd" in c.lower()]
    if lag_cols:
        group_scores["lags"] = max(abs(corrs[c]) for c in lag_cols)

    # Target encoding signal strength
    te_cols = [c for c in corrs if any(k in c.lower() for k in ["mean_sales", "dow_mean", "family_mean", "store_mean"])]
    if te_cols:
        group_scores["target_encoding"] = max(abs(corrs[c]) for c in te_cols)

    # Yearly signal strength
    yearly_cols = [c for c in corrs if "lag_364" in c or "lag_371" in c]
    if yearly_cols:
        group_scores["yearly"] = max(abs(corrs[c]) for c in yearly_cols)

    # Interactions signal strength
    interaction_cols = [c for c in corrs if any(k in c.lower() for k in ["promo_x_", "store_family", "promo_lift", "promo_ratio"])]
    if interaction_cols:
        group_scores["interactions"] = max(abs(corrs[c]) for c in interaction_cols)

    signals["group_priority"] = dict(
        sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    )

    return signals
