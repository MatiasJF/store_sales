"""
Generate final predictions and submission file.
"""

import numpy as np
import pandas as pd
from config import OUTPUT_DIR, ID_COL


def generate_submission(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Generate submission CSV from test data."""
    test_df = df[~df["is_train"]].copy()

    X_test = test_df[feature_cols].values
    preds = np.expm1(model.predict(X_test))
    preds = np.clip(preds, 0, None)

    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        "sales": preds,
    })
    submission = submission.sort_values(ID_COL).reset_index(drop=True)

    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)

    return submission
