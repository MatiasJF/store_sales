import numpy as np
from processing.features import register
from config import DATE_COL


@register(name="basic_date", group="temporal")
def basic_date(df):
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["day_of_month"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["year"] = df[DATE_COL].dt.year
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


@register(name="payday", group="temporal")
def payday(df):
    """Distance to nearest payday (15th and last day of month)."""
    day = df[DATE_COL].dt.day
    last_day = df[DATE_COL].dt.days_in_month

    dist_to_15 = (15 - day).abs()
    dist_to_end = (last_day - day).abs()
    df["days_to_payday"] = np.minimum(dist_to_15, dist_to_end)
    df["is_payday"] = ((day == 15) | (day == last_day)).astype(int)
    return df


@register(name="cyclic_date", group="temporal")
def cyclic_date(df):
    """Sine/cosine encoding of cyclical date features."""
    dow = df[DATE_COL].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    dom = df[DATE_COL].dt.day
    df["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    df["dom_cos"] = np.cos(2 * np.pi * dom / 31)

    month = df[DATE_COL].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


@register(name="year_progress", group="temporal")
def year_progress(df):
    df["day_of_year"] = df[DATE_COL].dt.dayofyear
    df["year_progress"] = df["day_of_year"] / 365.25
    return df
