"""
Autonomous feature search and hyperparameter optimization.
"""

import time
import numpy as np
import optuna
from models.train import train_and_evaluate
from optimizer.tracker import ExperimentTracker
from processing.features import get_groups, apply_features
from config import (
    TIME_BUDGET, OPTIMIZER_CONFIG, MODEL_TIERS, SEED,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Columns that are train-only or meta - exclude from feature set
_EXCLUDE_COLS = {"id", "sales", "is_train", "transactions"}


def feature_search(
    df,
    base_features: list[str],
    signal_priority: dict[str, float],
    tracker: ExperimentTracker,
) -> tuple[list[str], object]:
    """
    Greedy forward feature group selection guided by signal priority.
    Returns (best_feature_cols, enriched_df).
    """
    print("\n--- Phase 3: Feature Search ---")
    start = time.time()

    groups = get_groups()
    ordered_groups = sorted(
        groups.keys(),
        key=lambda g: signal_priority.get(g, 0),
        reverse=True,
    )

    current_features = list(base_features)
    best_score = float("inf")

    result = train_and_evaluate(df, current_features, model_name="lightgbm")
    best_score = result["score"]
    current_features = result["feature_cols"]
    tracker.log({
        "phase": "feature_search",
        "action": "baseline",
        "features": current_features,
        "score": best_score,
        "model_name": "lightgbm",
        "elapsed": result["elapsed"],
    })
    print(f"  Base score: {best_score:.5f} ({len(current_features)} features)")

    patience_counter = 0

    for group_name in ordered_groups:
        if time.time() - start > TIME_BUDGET["max_feature_search"]:
            print(f"  Feature search time budget exceeded.")
            break

        if patience_counter >= OPTIMIZER_CONFIG["feature_search_patience"]:
            print(f"  No improvement for {patience_counter} groups, stopping.")
            break

        feature_names = groups[group_name]
        try:
            df_trial = apply_features(df.copy(), feature_names)
        except Exception as e:
            print(f"  Skipping group '{group_name}': {e}")
            continue

        new_cols = [
            c for c in df_trial.select_dtypes(include=[np.number]).columns
            if c not in current_features
            and c not in _EXCLUDE_COLS
        ]

        if not new_cols:
            continue

        trial_features = current_features + new_cols
        result = train_and_evaluate(df_trial, trial_features, model_name="lightgbm")

        tracker.log({
            "phase": "feature_search",
            "action": f"add_group_{group_name}",
            "new_features": new_cols,
            "total_features": len(trial_features),
            "score": result["score"],
            "model_name": "lightgbm",
            "elapsed": result["elapsed"],
        })

        if result["score"] < best_score:
            improvement = (best_score - result["score"]) / best_score * 100
            print(f"  + {group_name}: {result['score']:.5f} ({improvement:+.2f}%) -> KEEP")
            best_score = result["score"]
            current_features = trial_features
            df = df_trial
            patience_counter = 0
        else:
            print(f"  + {group_name}: {result['score']:.5f} -> SKIP")
            patience_counter += 1

    print(f"  Best after feature search: {best_score:.5f} ({len(current_features)} features)")
    return current_features, df


def hyperparam_search(
    df,
    feature_cols: list[str],
    model_name: str,
    tracker: ExperimentTracker,
    n_trials: int | None = None,
) -> dict:
    """
    Optuna hyperparameter search for a given model.
    Returns best params found.
    """
    n_trials = n_trials or OPTIMIZER_CONFIG["optuna_n_trials"]
    print(f"\n--- Phase 4: Hyperparameter Tuning ({model_name}, {n_trials} trials) ---")
    start = time.time()
    best_score = float("inf")
    best_params = {}

    def objective(trial):
        nonlocal best_score, best_params

        if time.time() - start > TIME_BUDGET["max_hyperparam_search"]:
            raise optuna.TrialPruned()

        if model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

        result = train_and_evaluate(df, feature_cols, model_name=model_name, params=params)

        tracker.log({
            "phase": "hyperparam_search",
            "model_name": model_name,
            "params": params,
            "score": result["score"],
            "elapsed": result["elapsed"],
        })

        if result["score"] < best_score:
            best_score = result["score"]
            best_params = params

        return result["score"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except Exception:
        pass

    print(f"  Best score: {best_score:.5f}")
    print(f"  Trials completed: {len(study.trials)}")
    return best_params, best_score


def tier2_test(
    df,
    feature_cols: list[str],
    tier1_score: float,
    tracker: ExperimentTracker,
) -> tuple[str, dict, float]:
    """
    Quick test of Tier 2 models. If any beats Tier 1 by threshold, return it.
    """
    print("\n--- Tier 2 Model Comparison ---")
    best_model = "lightgbm"
    best_score = tier1_score
    best_params = {}
    threshold = OPTIMIZER_CONFIG["tier2_improvement_threshold"]

    for model_name in MODEL_TIERS["tier2"]:
        try:
            result = train_and_evaluate(df, feature_cols, model_name=model_name)
            tracker.log({
                "phase": "tier2_test",
                "model_name": model_name,
                "score": result["score"],
                "elapsed": result["elapsed"],
            })

            improvement = (tier1_score - result["score"]) / tier1_score
            status = "BETTER" if improvement > threshold else "similar"
            print(f"  {model_name}: {result['score']:.5f} ({improvement*100:+.2f}%) [{status}]")

            if result["score"] < best_score and improvement > threshold:
                best_model = model_name
                best_score = result["score"]
        except Exception as e:
            print(f"  {model_name}: FAILED ({e})")

    if best_model != "lightgbm":
        print(f"  -> Switching to {best_model} for hyperparameter tuning")
    else:
        print(f"  -> Sticking with lightgbm")

    return best_model, best_params, best_score
