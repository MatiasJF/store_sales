import numpy as np
from processing.features import register
from config import STORE_COL, FAMILY_COL

_GROUP_COLS = [STORE_COL, FAMILY_COL]


@register(name="promo_basic", group="promotions")
def promo_basic(df):
    """Basic promotion features."""
    df["has_promo"] = (df["onpromotion"] > 0).astype(int)
    return df


@register(name="promo_rolling", group="promotions")
def promo_rolling(df):
    """Rolling promotion counts."""
    df["promo_rmean_7"] = (
        df.groupby(_GROUP_COLS)["onpromotion"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["promo_rmean_14"] = (
        df.groupby(_GROUP_COLS)["onpromotion"]
        .transform(lambda x: x.rolling(14, min_periods=1).mean())
    )
    return df


@register(name="promo_change", group="promotions")
def promo_change(df):
    """Promotion change vs previous day."""
    df["promo_diff"] = df.groupby(_GROUP_COLS)["onpromotion"].diff().fillna(0)
    return df
