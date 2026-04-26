"""
Generate final predictions and submission file.
"""

import numpy as np
import pandas as pd
from config import OUTPUT_DIR, ID_COL


def _apply_zero_postprocessing(test_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Suppress predictions for store/family combos that historically sell ~zero."""
    if "zero_rate_hist" not in test_df.columns:
        return preds
    zr = test_df["zero_rate_hist"].values
    preds = preds.copy()
    # Force to 0 where zero_rate > 95%
    preds[zr > 0.95] = 0.0
    # Shrink where zero_rate > 80%
    shrink_mask = (zr > 0.80) & (zr <= 0.95)
    preds[shrink_mask] *= (1.0 - zr[shrink_mask])
    return preds


def generate_submission(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    models_for_ensemble: list[tuple] | None = None,
) -> pd.DataFrame:
    """Generate submission CSV from test data.

    If models_for_ensemble is provided as [(model, features, cv_score), ...],
    blend predictions using inverse-CV-score weights.
    """
    test_df = df[~df["is_train"]].copy()

    if models_for_ensemble and len(models_for_ensemble) > 1:
        # Ensemble: weighted average by inverse CV score
        all_preds = []
        weights = []
        for m, feats, cv_score in models_for_ensemble:
            X = test_df[feats].values
            p = np.expm1(m.predict(X))
            p = np.clip(p, 0, None)
            all_preds.append(p)
            weights.append(1.0 / max(cv_score, 1e-8))

        weights = np.array(weights) / sum(weights)
        preds = sum(w * p for w, p in zip(weights, all_preds))
    else:
        X_test = test_df[feature_cols].values
        preds = np.expm1(model.predict(X_test))
        preds = np.clip(preds, 0, None)

    preds = _apply_zero_postprocessing(test_df, preds)

    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        "sales": preds,
    })
    submission = submission.sort_values(ID_COL).reset_index(drop=True)

    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)

    return submission
