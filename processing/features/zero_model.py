import numpy as np
from processing.features import register
from config import STORE_COL, FAMILY_COL, TARGET, FORECAST_HORIZON

_GROUP_COLS = [STORE_COL, FAMILY_COL]
_HORIZON = FORECAST_HORIZON


@register(name="zero_rate_hist", group="zero_handling")
def zero_rate_hist(df):
    """Historical zero-sales rate per store/family, shifted by horizon."""
    is_zero = (df[TARGET] == 0).astype(float)
    df["zero_rate_hist"] = (
        df.groupby(_GROUP_COLS)[TARGET]
        .transform(lambda x: (x == 0).astype(float).shift(_HORIZON).expanding(min_periods=1).mean())
    )
    return df
