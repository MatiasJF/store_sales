import pandas as pd
import numpy as np
from config import DATE_COL, STORE_COL


def _merge_stores(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    return df.merge(stores, on=STORE_COL, how="left")


def _merge_oil(df: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    return df.merge(oil, on=DATE_COL, how="left")


def _build_holiday_features(holidays: pd.DataFrame) -> pd.DataFrame:
    """Build a date-level holiday lookup with national, regional, local flags."""
    nat = holidays[
        (holidays["locale"] == "National") & (holidays["is_real_holiday"])
    ].copy()
    nat = nat.groupby(DATE_COL).agg(
        is_national_holiday=("type", "count"),
        national_holiday_type=("type", "first"),
    ).reset_index()
    nat["is_national_holiday"] = 1

    regional = holidays[
        (holidays["locale"] == "Regional") & (holidays["is_real_holiday"])
    ].copy()
    regional = regional.groupby([DATE_COL, "locale_name"]).agg(
        is_regional_holiday=("type", "count"),
    ).reset_index()
    regional["is_regional_holiday"] = 1

    local = holidays[
        (holidays["locale"] == "Local") & (holidays["is_real_holiday"])
    ].copy()
    local = local.groupby([DATE_COL, "locale_name"]).agg(
        is_local_holiday=("type", "count"),
    ).reset_index()
    local["is_local_holiday"] = 1

    # Work days (negative holidays — people work on these)
    work_days = holidays[holidays["type"] == "Work Day"][
        [DATE_COL]
    ].drop_duplicates()
    work_days["is_work_day"] = 1

    return {
        "national": nat[[DATE_COL, "is_national_holiday", "national_holiday_type"]],
        "regional": regional[[DATE_COL, "locale_name", "is_regional_holiday"]],
        "local": local[[DATE_COL, "locale_name", "is_local_holiday"]],
        "work_days": work_days,
    }


def _merge_holidays(
    df: pd.DataFrame, holiday_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    # National holidays: join on date only
    df = df.merge(
        holiday_tables["national"], on=DATE_COL, how="left"
    )
    df["is_national_holiday"] = df["is_national_holiday"].fillna(0).astype(int)
    df["national_holiday_type"] = df["national_holiday_type"].fillna("none")

    # Regional holidays: join on date + state
    df = df.merge(
        holiday_tables["regional"],
        left_on=[DATE_COL, "state"],
        right_on=[DATE_COL, "locale_name"],
        how="left",
    )
    df["is_regional_holiday"] = df["is_regional_holiday"].fillna(0).astype(int)
    df.drop(columns=["locale_name"], inplace=True, errors="ignore")

    # Local holidays: join on date + city
    df = df.merge(
        holiday_tables["local"],
        left_on=[DATE_COL, "city"],
        right_on=[DATE_COL, "locale_name"],
        how="left",
    )
    df["is_local_holiday"] = df["is_local_holiday"].fillna(0).astype(int)
    df.drop(columns=["locale_name"], inplace=True, errors="ignore")

    # Work days
    df = df.merge(holiday_tables["work_days"], on=DATE_COL, how="left")
    df["is_work_day"] = df["is_work_day"].fillna(0).astype(int)

    # Combined holiday flag
    df["is_holiday"] = (
        (df["is_national_holiday"] == 1)
        | (df["is_regional_holiday"] == 1)
        | (df["is_local_holiday"] == 1)
    ).astype(int)

    return df


def _merge_transactions(df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Merge transactions (train-only, will be NaN for test)."""
    return df.merge(transactions, on=[DATE_COL, STORE_COL], how="left")


def build_base_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge train + test vertically, then join all supplementary sources.
    Returns a single DataFrame with a 'is_train' flag.
    """
    train = data["train"].copy()
    test = data["test"].copy()

    train["is_train"] = True
    test["is_train"] = False
    test["sales"] = np.nan

    df = pd.concat([train, test], ignore_index=True)

    # Join store metadata
    df = _merge_stores(df, data["stores"])

    # Join oil prices
    df = _merge_oil(df, data["oil"])
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    # Build and join holiday features
    holiday_tables = _build_holiday_features(data["holidays"])
    df = _merge_holidays(df, holiday_tables)

    # Join transactions
    df = _merge_transactions(df, data["transactions"])

    # Sort for time-series consistency
    df = df.sort_values([STORE_COL, "family", DATE_COL]).reset_index(drop=True)

    return df
