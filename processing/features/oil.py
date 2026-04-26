import numpy as np
from processing.features import register
from config import DATE_COL


@register(name="oil_lags", group="oil")
def oil_lags(df):
    """Oil price lagged by 7, 14, 30 days."""
    for lag in [7, 14, 30]:
        df[f"oil_lag_{lag}"] = df["dcoilwtico"].shift(lag)
    return df


@register(name="oil_rolling", group="oil")
def oil_rolling(df):
    """Oil rolling mean and std."""
    df["oil_rmean_7"] = df["dcoilwtico"].rolling(7, min_periods=1).mean()
    df["oil_rmean_30"] = df["dcoilwtico"].rolling(30, min_periods=1).mean()
    df["oil_rstd_30"] = df["dcoilwtico"].rolling(30, min_periods=2).std()
    return df


@register(name="oil_change", group="oil")
def oil_change(df):
    """Oil price percentage change."""
    df["oil_pct_change_7"] = df["dcoilwtico"].pct_change(7)
    df["oil_pct_change_30"] = df["dcoilwtico"].pct_change(30)
    return df
