# Store Sales — Kaggle Competition

## Current State (2026-04-27)

**Branch**: `improve/v2-creative-gains`
**Last working score**: 0.4357 (branch `improve/score-optimization`, commit `12d9802`)
**Broken score**: 2.5934 (V2 — per-family blending bug)

## CRITICAL BUG TO FIX

The per-family model blending in `pipeline.py` (lines ~294-328) has an **index mismatch**:
- `generate_submission()` returns predictions sorted by ID
- `test_df = df[~df["is_train"]]` uses original df row order
- The blend mixes these two orderings, scrambling predictions across families
- Result: BOOKS (should be ~0) gets predictions of ~234, causing RMSLE explosion

### Fix
Restructure the blending to work on ID-sorted data. Change `_train_family_models` to return family labels instead of masks. Then in the blending loop, filter by family on the ID-sorted test_df.

## What Works in V2 (keep after fixing bug)
- **Zero classifier (p_zero feature)**: CV 0.3963 → 0.3903 — biggest single feature gain
- **Transaction proxy**: predicted_transactions from model trained on transactions.csv
- **Temporal sample weighting**: exp decay, half-life 180 days in `train.py`
- **Per-family models**: GROCERY I CV=0.156, PRODUCE=0.145 (vs global 0.389)

## Score History
| Version | CV | Kaggle LB |
|---|---|---|
| V1 (commit 12d9802) | 0.3967 | 0.4357 |
| V2 (BROKEN) | 0.3887 | 2.5934 |

## Key Data Facts
- 3M train rows, 28,512 test rows (16 days × 54 stores × 33 families)
- 31% zero sales; BOOKS 97% zeros
- Top 5 families = 81% of total sales
- RMSLE heavily penalizes predicting positive when actual is 0
- CV-LB gap was ~10% in V1 (0.397 CV vs 0.436 LB)

## Architecture
```
pipeline.py          — Main orchestrator
models/train.py      — CV (5-fold, gap=16) + sample weights + training
models/predict.py    — Submission gen + zero post-processing
models/zero_classifier.py  — P(sales=0) binary classifier
models/transaction_proxy.py — Predict transactions for test period
optimizer/search.py  — Feature search + Optuna
processing/features/ — Feature registry with @register decorator
config.py           — n_splits=5, gap=16, TRAIN_START_DATE="2015-08-01"
```
