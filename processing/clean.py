import pandas as pd
import numpy as np


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sales"] = df["sales"].clip(lower=0)
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)
    return df


def clean_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)
    return df


def clean_oil(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("date").resample("D").first().reset_index()
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    return df


def clean_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Mark transferred holidays as non-holidays (they were moved)
    df["is_real_holiday"] = ~df["transferred"]
    return df


def clean_stores(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def clean_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply cleaning to all dataframes."""
    return {
        "train": clean_train(data["train"]),
        "test": clean_test(data["test"]),
        "stores": clean_stores(data["stores"]),
        "oil": clean_oil(data["oil"]),
        "holidays": clean_holidays(data["holidays"]),
        "transactions": clean_transactions(data["transactions"]),
        "sample_submission": data["sample_submission"],
    }
