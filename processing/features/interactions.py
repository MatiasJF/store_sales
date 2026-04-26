import numpy as np
from processing.features import register
from config import STORE_COL, FAMILY_COL, TARGET, FORECAST_HORIZON

_GROUP_COLS = [STORE_COL, FAMILY_COL]
_HORIZON = FORECAST_HORIZON


@register(name="store_day_promo_count", group="interactions")
def store_day_promo_count(df):
    """Total promo count across all families for same store+date (traffic proxy)."""
    promo_counts = df.groupby([STORE_COL, "date"])["onpromotion"].transform("sum")
    df["store_day_promo_count"] = promo_counts
    return df


@register(name="promo_holiday_interaction", group="interactions")
def promo_holiday_interaction(df):
    """Interaction between promotions and holidays."""
    df["promo_x_holiday"] = df["has_promo"] * df["is_holiday"] if "has_promo" in df.columns else 0
    df["promo_x_weekend"] = df["has_promo"] * df["is_weekend"] if "has_promo" in df.columns else 0
    return df


@register(name="promo_x_payday", group="interactions")
def promo_x_payday(df):
    """Interaction between promotions and payday."""
    if "has_promo" in df.columns and "is_payday" in df.columns:
        df["promo_x_payday"] = df["has_promo"] * df["is_payday"]
    return df


@register(name="promo_sales_lift", group="interactions")
def promo_sales_lift(df):
    """Historical promo ratio and promo vs non-promo sales lift per family."""
    # Promo ratio: fraction of time on promotion (shifted by horizon)
    df["promo_ratio"] = (
        df.groupby(_GROUP_COLS)["onpromotion"]
        .transform(lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean())
    )

    # Promo sales lift: ratio of avg promo sales to avg non-promo sales per store/family
    # Use expanding shifted stats
    promo_mask = df["onpromotion"] > 0
    df["_promo_sales"] = np.where(promo_mask, df[TARGET], np.nan)
    df["_nopromo_sales"] = np.where(~promo_mask, df[TARGET], np.nan)

    df["_promo_avg"] = (
        df.groupby(_GROUP_COLS)["_promo_sales"]
        .transform(lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean())
    )
    df["_nopromo_avg"] = (
        df.groupby(_GROUP_COLS)["_nopromo_sales"]
        .transform(lambda x: x.shift(_HORIZON).expanding(min_periods=1).mean())
    )

    df["promo_lift"] = (df["_promo_avg"] / df["_nopromo_avg"].replace(0, np.nan)).fillna(1.0)
    df.drop(columns=["_promo_sales", "_nopromo_sales", "_promo_avg", "_nopromo_avg"], inplace=True)
    return df
