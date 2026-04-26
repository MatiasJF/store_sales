import numpy as np
from processing.features import register
from config import DATE_COL


@register(name="holiday_proximity", group="holidays")
def holiday_proximity(df):
    """Days until/since nearest national holiday."""
    holiday_dates = df.loc[df["is_national_holiday"] == 1, DATE_COL].unique()
    holiday_dates = np.sort(holiday_dates)

    dates = df[DATE_COL].values
    idx = np.searchsorted(holiday_dates, dates)

    # Days to next holiday
    next_idx = np.clip(idx, 0, len(holiday_dates) - 1)
    days_to_next = (holiday_dates[next_idx] - dates).astype("timedelta64[D]").astype(float)
    days_to_next = np.where(days_to_next < 0, 999, days_to_next)

    # Days since last holiday
    prev_idx = np.clip(idx - 1, 0, len(holiday_dates) - 1)
    days_since_last = (dates - holiday_dates[prev_idx]).astype("timedelta64[D]").astype(float)
    days_since_last = np.where(days_since_last < 0, 999, days_since_last)

    df["days_to_next_holiday"] = days_to_next
    df["days_since_last_holiday"] = days_since_last
    return df


@register(name="holiday_window", group="holidays")
def holiday_window(df):
    """Binary flags for being within N days of a holiday."""
    for window in [1, 3, 7]:
        df[f"near_holiday_{window}d"] = (
            (df["days_to_next_holiday"] <= window)
            | (df["days_since_last_holiday"] <= window)
        ).astype(int)
    return df


@register(name="earthquake", group="holidays")
def earthquake(df):
    """Earthquake impact: 2016-04-16, with exponential decay over weeks."""
    eq_date = np.datetime64("2016-04-16")
    days_since = (df[DATE_COL].values - eq_date).astype("timedelta64[D]").astype(float)
    # Active only for 0-60 days after earthquake
    in_window = (days_since >= 0) & (days_since <= 60)
    df["earthquake_impact"] = np.where(in_window, np.exp(-days_since / 14), 0)
    return df
