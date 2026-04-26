import numpy as np
from processing.features import register
from config import STORE_COL, FAMILY_COL, TARGET, FORECAST_HORIZON

_GROUP_COLS = [STORE_COL, FAMILY_COL]
_HORIZON = FORECAST_HORIZON  # minimum safe shift to avoid future leakage


def _lag(df, col, periods, prefix=None):
    prefix = prefix or col
    name = f"{prefix}_lag_{periods}"
    df[name] = df.groupby(_GROUP_COLS)[col].shift(periods)
    return df, name


@register(name="sales_lags_safe", group="lags")
def sales_lags_safe(df):
    """Horizon-safe lags: only lags >= FORECAST_HORIZON."""
    for lag in [_HORIZON, 21, 28, 35, 42]:
        df, _ = _lag(df, TARGET, lag)
    return df


@register(name="rolling_mean_safe_28", group="lags")
def rolling_mean_safe_28(df):
    """Rolling mean over 28 days, shifted by horizon to prevent leakage."""
    df["sales_rmean_28"] = (
        df.groupby(_GROUP_COLS)[TARGET]
        .transform(lambda x: x.shift(_HORIZON).rolling(28, min_periods=1).mean())
    )
    return df


@register(name="rolling_mean_safe_14", group="lags")
def rolling_mean_safe_14(df):
    """Rolling mean over 14 days, shifted by horizon."""
    df["sales_rmean_14"] = (
        df.groupby(_GROUP_COLS)[TARGET]
        .transform(lambda x: x.shift(_HORIZON).rolling(14, min_periods=1).mean())
    )
    return df


@register(name="rolling_std_safe_28", group="lags")
def rolling_std_safe_28(df):
    """Rolling std over 28 days, shifted by horizon."""
    df["sales_rstd_28"] = (
        df.groupby(_GROUP_COLS)[TARGET]
        .transform(lambda x: x.shift(_HORIZON).rolling(28, min_periods=2).std())
    )
    return df


@register(name="rolling_mean_safe_60", group="lags")
def rolling_mean_safe_60(df):
    """Rolling mean over 60 days, shifted by horizon."""
    df["sales_rmean_60"] = (
        df.groupby(_GROUP_COLS)[TARGET]
        .transform(lambda x: x.shift(_HORIZON).rolling(60, min_periods=1).mean())
    )
    return df


@register(name="expanding_mean_safe", group="lags")
def expanding_mean_safe(df):
    """Expanding mean shifted by horizon."""
    df["sales_exp_mean"] = (
        df.groupby(_GROUP_COLS)[TARGET]
        .transform(lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean())
    )
    return df


@register(name="sales_momentum", group="lags")
def sales_momentum(df):
    """Momentum: ratio of recent (14d) to long-term (60d) rolling mean."""
    rmean_14 = df.groupby(_GROUP_COLS)[TARGET].transform(
        lambda x: x.shift(_HORIZON).rolling(14, min_periods=1).mean()
    )
    rmean_60 = df.groupby(_GROUP_COLS)[TARGET].transform(
        lambda x: x.shift(_HORIZON).rolling(60, min_periods=1).mean()
    )
    df["sales_momentum"] = (rmean_14 / rmean_60.replace(0, np.nan)).fillna(1.0)
    return df
