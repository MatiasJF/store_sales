import pandas as pd
from config import DATA_FILES


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILES["train"], parse_dates=["date"])
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILES["test"], parse_dates=["date"])
    return df


def load_stores() -> pd.DataFrame:
    return pd.read_csv(DATA_FILES["stores"])


def load_oil() -> pd.DataFrame:
    return pd.read_csv(DATA_FILES["oil"], parse_dates=["date"])


def load_holidays() -> pd.DataFrame:
    return pd.read_csv(DATA_FILES["holidays"], parse_dates=["date"])


def load_transactions() -> pd.DataFrame:
    return pd.read_csv(DATA_FILES["transactions"], parse_dates=["date"])


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_FILES["sample_submission"])


def load_all_competition_data() -> dict[str, pd.DataFrame]:
    """Load all competition CSVs into a dictionary."""
    return {
        "train": load_train(),
        "test": load_test(),
        "stores": load_stores(),
        "oil": load_oil(),
        "holidays": load_holidays(),
        "transactions": load_transactions(),
        "sample_submission": load_sample_submission(),
    }
