import os
from pathlib import Path

# -- Paths --
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# -- Data files --
DATA_FILES = {
    "train": DATA_DIR / "train.csv",
    "test": DATA_DIR / "test.csv",
    "stores": DATA_DIR / "stores.csv",
    "oil": DATA_DIR / "oil.csv",
    "holidays": DATA_DIR / "holidays_events.csv",
    "transactions": DATA_DIR / "transactions.csv",
    "sample_submission": DATA_DIR / "sample_submission.csv",
}

# -- Time budgets (seconds) --
TIME_BUDGET = {
    "max_per_model_cv": 120,
    "max_feature_search": 900,
    "max_hyperparam_search": 900,
    "max_total": 3600,
}

# -- Model tiers --
MODEL_TIERS = {
    "tier1": ["lightgbm"],
    "tier2": ["xgboost", "catboost"],
    "tier3": ["ensemble"],
}

# -- Forecast horizon --
FORECAST_HORIZON = 16  # 16 dates in test (Aug 16-31)

# -- Training data window --
TRAIN_START_DATE = "2015-08-01"  # ~2 years of data; drop stale early history

# -- Cross-validation --
CV_CONFIG = {
    "n_splits": 5,
    "forecast_horizon": FORECAST_HORIZON,
    "gap": 16,  # simulate real forecast gap between train cutoff and val start
}

# -- Optimizer --
OPTIMIZER_CONFIG = {
    "feature_search_patience": 5,  # stop after N rounds with no improvement
    "optuna_n_trials": 50,
    "tier2_improvement_threshold": 0.005,  # 0.5% relative improvement to justify tier2
}

# -- Target --
TARGET = "sales"
ID_COL = "id"
DATE_COL = "date"
STORE_COL = "store_nbr"
FAMILY_COL = "family"

# -- Random seed --
SEED = 42
