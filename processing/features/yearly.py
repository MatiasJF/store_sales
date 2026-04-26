from processing.features import register
from config import STORE_COL, FAMILY_COL, TARGET

_GROUP_COLS = [STORE_COL, FAMILY_COL]


@register(name="yearly_lags", group="yearly")
def yearly_lags(df):
    """Year-over-year lags: 364 and 371 days (same weekday, last year)."""
    for lag in [364, 371]:
        name = f"{TARGET}_lag_{lag}"
        df[name] = df.groupby(_GROUP_COLS)[TARGET].shift(lag)
    return df
