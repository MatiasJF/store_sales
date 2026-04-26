import numpy as np
import pandas as pd
from processing.features import register
from config import STORE_COL, FAMILY_COL, TARGET, DATE_COL, FORECAST_HORIZON

_GROUP_COLS = [STORE_COL, FAMILY_COL]
_HORIZON = FORECAST_HORIZON


@register(name="hist_agg_28d", group="target_encoding")
def hist_agg_28d(df):
    """Per store/family: mean and std of sales in last 28 days (shifted by horizon)."""
    g = df.groupby(_GROUP_COLS)[TARGET]
    df["mean_sales_last_28d"] = g.transform(
        lambda x: x.shift(_HORIZON).rolling(28, min_periods=1).mean()
    )
    df["std_sales_last_28d"] = g.transform(
        lambda x: x.shift(_HORIZON).rolling(28, min_periods=2).std()
    )
    return df


@register(name="hist_agg_90d", group="target_encoding")
def hist_agg_90d(df):
    """Per store/family: mean sales in last 90 days (shifted by horizon)."""
    df["mean_sales_last_90d"] = (
        df.groupby(_GROUP_COLS)[TARGET]
        .transform(lambda x: x.shift(_HORIZON).rolling(90, min_periods=1).mean())
    )
    return df


@register(name="dow_profile", group="target_encoding")
def dow_profile(df):
    """Average sales by store/family/day-of-week. Uses expanding mean shifted by horizon."""
    dow = df[DATE_COL].dt.dayofweek
    group_keys = [STORE_COL, FAMILY_COL, dow]

    # Build expanding mean per store/family/dow, shifted by horizon
    df["_dow"] = dow
    df["dow_mean_sales"] = (
        df.groupby([STORE_COL, FAMILY_COL, "_dow"])[TARGET]
        .transform(lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean())
    )
    df.drop(columns=["_dow"], inplace=True)
    return df


@register(name="store_family_target_enc", group="target_encoding")
def store_family_target_enc(df):
    """Historical mean sales per store, per family, per store-family (shifted by horizon)."""
    g_store = df.groupby(STORE_COL)[TARGET]
    df["store_mean_sales"] = g_store.transform(
        lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean()
    )

    g_family = df.groupby(FAMILY_COL)[TARGET]
    df["family_mean_sales"] = g_family.transform(
        lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean()
    )

    g_sf = df.groupby(_GROUP_COLS)[TARGET]
    df["store_family_mean_sales"] = g_sf.transform(
        lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean()
    )
    return df
