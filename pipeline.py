"""
Autonomous pipeline orchestrator.
Run once -> iterates through all phases -> produces submission.csv
"""

import time
import json
import numpy as np
import pandas as pd

from config import (
    TARGET, OUTPUT_DIR, TIME_BUDGET, MODEL_TIERS,
)
from sources import load_all_competition_data
from processing.clean import clean_all
from processing.merge import build_base_table
from processing.features import apply_features, get_feature_names, get_groups
from insights.signals import detect_signals
from insights.scoring import score_features
from models.train import train_and_evaluate, train_final_model
from models.predict import generate_submission
from optimizer.search import feature_search, hyperparam_search, tier2_test
from optimizer.tracker import ExperimentTracker

# Columns that are train-only (NaN for test) and should be excluded from features
_EXCLUDE_COLS = {"id", TARGET, "is_train", "transactions"}


def _get_numeric_features(df):
    """Get numeric feature columns, excluding known-bad ones."""
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _EXCLUDE_COLS
    ]


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

    print("\n  Seasonality signals:")
    for k, v in signals["seasonality"].items():
        print(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")

    print("\n  Oil lag correlations:")
    for k, v in signals["oil_lags"].items():
        print(f"    {k}: {v:.4f}")

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
    # PHASE 5: FINAL OUTPUT
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Final Model & Submission")
    print("=" * 60)

    t0 = time.time()
    final_model, final_features = train_final_model(
        df, best_features, model_name=best_model, params=best_params,
    )
    submission = generate_submission(df, final_model, final_features)
    print(f"  Final model trained on {len(final_features)} features ({time.time()-t0:.1f}s)")
    print(f"  Submission: {len(submission)} rows")
    print(f"  Predictions range: {submission['sales'].min():.2f} - {submission['sales'].max():.2f}")
    print(f"  Predictions mean: {submission['sales'].mean():.2f}")

    # Save best config
    best_config = {
        "model_name": best_model,
        "params": best_params,
        "features": final_features,
        "baseline_score": baseline_score,
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
