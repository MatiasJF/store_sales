"""
Autonomous pipeline orchestrator.
Run once -> iterates through all phases -> produces submission.csv

V2: Two-stage zero classifier, transaction proxy, temporal weighting,
    per-family models, ensemble blending.
"""

import time
import json
import numpy as np
import pandas as pd

from config import (
    TARGET, OUTPUT_DIR, TIME_BUDGET, MODEL_TIERS,
    STORE_COL, FAMILY_COL, DATE_COL, TRAIN_START_DATE,
)
from sources import load_all_competition_data
from processing.clean import clean_all
from processing.merge import build_base_table
from processing.features import apply_features, get_feature_names, get_groups
from insights.signals import detect_signals
from insights.scoring import score_features
from models.train import train_and_evaluate, train_final_model
from models.predict import generate_submission
from models.zero_classifier import train_zero_classifier, add_zero_probability
from models.transaction_proxy import build_transaction_proxy
from optimizer.search import feature_search, hyperparam_search, tier2_test
from optimizer.tracker import ExperimentTracker

# Columns that are train-only (NaN for test) and should be excluded from features
_EXCLUDE_COLS = {"id", TARGET, "is_train", "transactions"}

# Low-value features to prune (correlation < 0.01 with target or redundant)
_LOW_VALUE_FEATURES = {
    "dom_sin", "dom_cos", "month_sin", "month_cos",
    "is_work_day", "is_payday",
    "near_holiday_7d",  # redundant with near_holiday_3d
    "oil_pct_change_7",
    "store_family_encoded",  # 1782-cardinality meaningless integer
}

# Family clusters for per-family modeling
# Top families each get their own model; rest share one
_TOP_FAMILIES = ["GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY"]


def _get_numeric_features(df):
    """Get numeric feature columns, excluding known-bad and low-value ones."""
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _EXCLUDE_COLS and c not in _LOW_VALUE_FEATURES
    ]


def _train_family_models(
    df: pd.DataFrame,
    best_features: list[str],
    best_params: dict,
    best_model: str,
) -> list[tuple]:
    """Train separate models for top families + one for the rest.
    Returns list of (family_mask_for_test, model, features, cv_score).
    """
    print("\n  Training per-family models...")
    family_models = []

    for family_group in _TOP_FAMILIES + ["_OTHER"]:
        if family_group == "_OTHER":
            mask = ~df[FAMILY_COL].isin(_TOP_FAMILIES)
            label = "OTHER (28 families)"
        else:
            mask = df[FAMILY_COL] == family_group
            label = family_group

        df_sub = df[mask].copy()
        if df_sub[df_sub["is_train"]].shape[0] < 1000:
            print(f"    {label}: too few rows, skipping")
            continue

        cv_result = train_and_evaluate(
            df_sub, best_features, model_name=best_model, params=best_params,
        )
        model, features = train_final_model(
            df_sub, best_features, model_name=best_model, params=best_params,
        )

        test_mask = ~df["is_train"] & mask
        family_models.append((test_mask, model, features, cv_result["score"]))
        print(f"    {label}: CV={cv_result['score']:.5f} ({features.__len__()} feats)")

    return family_models


def run_pipeline():
    pipeline_start = time.time()
    tracker = ExperimentTracker()

    # ================================================================
    # PHASE 1: FOUNDATION
    # ================================================================
    print("=" * 60)
    print("PHASE 1: Foundation - Load, Clean, Merge")
    print("=" * 60)

    t0 = time.time()
    raw_data = load_all_competition_data()
    clean_data = clean_all(raw_data)
    df = build_base_table(clean_data)
    print(f"  Base table: {df.shape[0]:,} rows x {df.shape[1]} cols ({time.time()-t0:.1f}s)")

    # ================================================================
    # PHASE 1a: TRANSACTION PROXY
    # ================================================================
    print("\n  Building transaction proxy...")
    t0 = time.time()
    try:
        df = build_transaction_proxy(df)
        print(f"  Transaction proxy added ({time.time()-t0:.1f}s)")
        print(f"    predicted_transactions range: {df['predicted_transactions'].min():.0f} - {df['predicted_transactions'].max():.0f}")
    except Exception as e:
        print(f"  Transaction proxy failed: {e}")

    # Apply ALL features first so signals can see them
    print("  Applying all features for signal detection...")
    all_feature_names = get_feature_names()
    try:
        df_full = apply_features(df.copy(), all_feature_names)
    except Exception as e:
        print(f"  Warning: Some features failed during full apply: {e}")
        df_full = df.copy()
        for group_name, feat_names in get_groups().items():
            try:
                df_full = apply_features(df_full, feat_names)
            except Exception as e2:
                print(f"    Skipping group '{group_name}': {e2}")

    # ================================================================
    # PHASE 1b: SIGNAL DETECTION
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1b: Signal Detection")
    print("=" * 60)

    t0 = time.time()
    signals = detect_signals(df_full)
    print(f"  Signal detection: {time.time()-t0:.1f}s")

    print("\n  Group priority (by signal strength):")
    for group, strength in signals["group_priority"].items():
        print(f"    {group}: {strength:.4f}")

    print("\n  Top 10 correlations with sales:")
    for col, corr in list(signals["correlations"].items())[:10]:
        print(f"    {col}: {corr:.4f}")

    # Save signals
    with open(OUTPUT_DIR / "signals.json", "w") as f:
        json.dump(
            {k: v for k, v in signals.items() if k != "zero_rates"},
            f, indent=2, default=str,
        )

    # ================================================================
    # PHASE 1c: FEATURE IMPORTANCE SCORING
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1c: Feature Importance Scoring (quick baseline)")
    print("=" * 60)

    t0 = time.time()
    numeric_cols = _get_numeric_features(df_full)
    importances = score_features(df_full, numeric_cols)
    print(f"  Scored {len(importances)} features ({time.time()-t0:.1f}s)")

    print("\n  Top 15 features by importance:")
    for col, imp in list(importances.items())[:15]:
        print(f"    {col}: {imp:.0f}")

    # ================================================================
    # PHASE 2: BASELINE
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Baseline Model")
    print("=" * 60)

    # Start with safe non-target-dependent features
    df = apply_features(df, ["basic_date", "promo_basic", "store_encoded", "store_cluster"])
    base_cols = _get_numeric_features(df)

    result = train_and_evaluate(df, base_cols, model_name="lightgbm")
    baseline_score = result["score"]
    tracker.log({
        "phase": "baseline",
        "model_name": "lightgbm",
        "features": base_cols,
        "score": baseline_score,
        "elapsed": result["elapsed"],
    })
    print(f"  Baseline RMSLE: {baseline_score:.5f} ({len(base_cols)} features, {result['elapsed']:.1f}s)")

    # ================================================================
    # PHASE 3: FEATURE SEARCH
    # ================================================================
    best_features, df = feature_search(
        df, base_cols, signals["group_priority"], tracker,
    )

    # ================================================================
    # PHASE 3a: ZERO CLASSIFIER
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3a: Zero Classifier")
    print("=" * 60)

    t0 = time.time()
    try:
        zero_model, zero_feats = train_zero_classifier(df, best_features)
        df = add_zero_probability(df, zero_model, zero_feats)
        # Add p_zero to feature set
        if "p_zero" not in best_features:
            best_features = best_features + ["p_zero"]
        # Evaluate with p_zero
        result_with_pz = train_and_evaluate(df, best_features, model_name="lightgbm")
        print(f"  Zero classifier trained ({time.time()-t0:.1f}s)")
        print(f"  CV with p_zero: {result_with_pz['score']:.5f}")
        print(f"  p_zero stats: mean={df['p_zero'].mean():.3f}, test_mean={df[~df['is_train']]['p_zero'].mean():.3f}")
    except Exception as e:
        print(f"  Zero classifier failed: {e}")

    # ================================================================
    # PHASE 3b: TIER 2 MODEL TEST
    # ================================================================
    tier1_result = train_and_evaluate(df, best_features, model_name="lightgbm")
    tier1_score = tier1_result["score"]

    best_model, _, _ = tier2_test(df, best_features, tier1_score, tracker)

    # ================================================================
    # PHASE 4: HYPERPARAMETER TUNING
    # ================================================================
    elapsed_total = time.time() - pipeline_start
    remaining_budget = TIME_BUDGET["max_total"] - elapsed_total
    if remaining_budget < 60:
        print("\n  Skipping hyperparam search (time budget low)")
        best_params = {}
    else:
        est_per_trial = 30
        max_trials = max(10, int(remaining_budget / est_per_trial))
        best_params, tuned_score = hyperparam_search(
            df, best_features, best_model, tracker, n_trials=min(max_trials, 50),
        )

    # ================================================================
    # PHASE 5: FINAL OUTPUT — Per-Family Models + Ensemble
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Final Models & Submission")
    print("=" * 60)

    t0 = time.time()

    # --- Strategy A: Global ensemble (LightGBM + XGBoost) ---
    final_model, final_features = train_final_model(
        df, best_features, model_name=best_model, params=best_params,
    )
    primary_cv = train_and_evaluate(df, best_features, model_name=best_model, params=best_params)
    primary_cv_score = primary_cv["score"]
    print(f"  Global {best_model}: CV={primary_cv_score:.5f}")

    secondary_model_name = "xgboost" if best_model == "lightgbm" else "lightgbm"
    try:
        secondary_cv = train_and_evaluate(df, best_features, model_name=secondary_model_name)
        secondary_cv_score = secondary_cv["score"]
        secondary_model, secondary_features = train_final_model(
            df, best_features, model_name=secondary_model_name,
        )
        print(f"  Global {secondary_model_name}: CV={secondary_cv_score:.5f}")
        models_for_ensemble = [
            (final_model, final_features, primary_cv_score),
            (secondary_model, secondary_features, secondary_cv_score),
        ]
    except Exception as e:
        print(f"  Secondary model failed ({e})")
        models_for_ensemble = None

    # --- Strategy B: Per-family models ---
    try:
        family_models = _train_family_models(df, best_features, best_params, best_model)
    except Exception as e:
        print(f"  Per-family models failed: {e}")
        family_models = None

    # --- Generate submission: blend global ensemble + per-family ---
    # Start with global ensemble predictions
    submission = generate_submission(
        df, final_model, final_features,
        models_for_ensemble=models_for_ensemble,
    )
    global_preds = submission["sales"].values.copy()

    # Override with per-family predictions where available
    if family_models:
        test_df = df[~df["is_train"]].copy()
        family_preds = np.full(len(test_df), np.nan)

        for test_mask, model, features, cv_score in family_models:
            idx = test_mask[test_mask].index
            test_rows = test_df.loc[idx]
            X = test_rows[features].values
            preds = np.expm1(model.predict(X))
            preds = np.clip(preds, 0, None)
            # Map back to submission order
            for i, orig_idx in enumerate(idx):
                pos = test_df.index.get_loc(orig_idx)
                family_preds[pos] = preds[i]

        # Blend: 60% per-family, 40% global (where per-family is available)
        has_family = ~np.isnan(family_preds)
        blended = global_preds.copy()
        blended[has_family] = 0.6 * family_preds[has_family] + 0.4 * global_preds[has_family]

        # Apply zero post-processing on blended
        from models.predict import _apply_zero_postprocessing
        blended = _apply_zero_postprocessing(test_df.reset_index(drop=True), blended)

        submission["sales"] = blended
        submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
        print(f"\n  Blended per-family (60%) + global (40%)")

    print(f"  Submission: {len(submission)} rows ({time.time()-t0:.1f}s)")
    print(f"  Predictions range: {submission['sales'].min():.2f} - {submission['sales'].max():.2f}")
    print(f"  Predictions mean: {submission['sales'].mean():.2f}")
    print(f"  Zero predictions: {(submission['sales'] < 0.5).sum()} ({(submission['sales'] < 0.5).mean():.1%})")

    # Save best config
    best_config = {
        "model_name": best_model,
        "params": best_params,
        "features": final_features,
        "baseline_score": baseline_score,
        "ensemble": bool(models_for_ensemble),
        "per_family": bool(family_models),
        "best_cv_score": tracker.best()["score"] if tracker.best() else None,
    }
    with open(OUTPUT_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, default=str)

    tracker.save()

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  {tracker.summary()}")
    print(f"\n  Output files:")
    print(f"    {OUTPUT_DIR / 'submission.csv'}")
    print(f"    {OUTPUT_DIR / 'best_config.json'}")
    print(f"    {OUTPUT_DIR / 'experiments.json'}")
    print(f"    {OUTPUT_DIR / 'signals.json'}")

    return submission
