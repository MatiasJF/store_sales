import pandas as pd
from processing.features import register
from config import STORE_COL


@register(name="store_encoded", group="store")
def store_encoded(df):
    """Label-encode categorical store metadata."""
    for col in ["city", "state", "type"]:
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].astype("category").cat.codes
    return df


@register(name="store_cluster", group="store")
def store_cluster(df):
    """Cluster as a feature (already numeric)."""
    # Already present from merge, just ensure it's int
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].fillna(0).astype(int)
    return df
