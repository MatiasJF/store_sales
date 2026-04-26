# Store Sales - Time Series Forecasting: Data Profile

## Objective
Predict 15 days of sales (2017-08-16 to 2017-08-31) for every store-family combination at Favorita stores in Ecuador.

**Metric:** Submission = 28,512 rows (54 stores x 33 families x 16 dates), predicting `sales`.

---

## Files Overview

| File | Rows | Columns | Size |
|---|---|---|---|
| train.csv | 3,000,888 | 6 | ~122 MB |
| test.csv | 28,512 | 5 | ~1 MB |
| stores.csv | 54 | 5 | ~1 KB |
| oil.csv | 1,218 | 2 | ~20 KB |
| holidays_events.csv | 350 | 6 | ~22 KB |
| transactions.csv | 83,488 | 3 | ~1.5 MB |
| sample_submission.csv | 28,512 | 2 | ~342 KB |

---

## train.csv (Target Data)

- **Grain:** 1 row = 1 store x 1 family x 1 date (fully dense: 54 x 33 x 1,684 = 3,000,888)
- **Date range:** 2013-01-01 to 2017-08-15 (~4.6 years)
- **No nulls**

| Column | Type | Description |
|---|---|---|
| id | int | Row identifier (0 to 3,000,887) |
| date | date | Sale date |
| store_nbr | int | Store ID (1-54) |
| family | str | Product category (33 unique) |
| sales | float | Total sales (target). Min=0, Median=11, Mean=358, Max=124,717 |
| onpromotion | int | Number of promoted items in that family/store/date |

**Key stats:**
- 31.3% of rows have **zero sales** (939,130 rows)
- No negative sales
- Heavy right skew (mean >> median), log transform likely beneficial
- Sales are fractional (products sold by weight)

### 33 Product Families
AUTOMOTIVE, BABY CARE, BEAUTY, BEVERAGES, BOOKS, BREAD/BAKERY, CELEBRATION, CLEANING, DAIRY, DELI, EGGS, FROZEN FOODS, GROCERY I, GROCERY II, HARDWARE, HOME AND KITCHEN I, HOME AND KITCHEN II, HOME APPLIANCES, HOME CARE, LADIESWEAR, LAWN AND GARDEN, LINGERIE, LIQUOR/WINE/BEER, MAGAZINES, MEATS, PERSONAL CARE, PET SUPPLIES, PLAYERS AND ELECTRONICS, POULTRY, PREPARED FOODS, PRODUCE, SCHOOL AND OFFICE SUPPLIES, SEAFOOD

---

## test.csv (Prediction Target)

- Same structure as train, minus the `sales` column
- **Date range:** 2017-08-16 to 2017-08-31 (15 forecast days, 16 unique dates)
- 28,512 rows = 54 stores x 33 families x 16 dates
- No nulls

---

## stores.csv (Store Metadata)

54 stores across Ecuador:
- **22 cities** in **16 states**
- **5 store types:** A (9 stores), B (7), C (14), D (16), E (5) — likely size/format
- **17 clusters** — grouping of similar stores

Top cities: Quito (18 stores), Guayaquil (8 stores)

---

## oil.csv (External Feature)

- 1,218 rows covering both train and test period (2013-01-01 to 2017-08-31)
- **43 null values** (missing trading days — weekends/holidays)
- Price range: $26.19 to $110.62, Mean: $67.71
- Ecuador is oil-dependent — this is a key macro feature

---

## holidays_events.csv (External Feature)

350 rows covering 2012-03-02 to 2017-12-26:

| Type | Count |
|---|---|
| Holiday | 221 |
| Event | 56 |
| Additional | 51 |
| Transfer | 12 |
| Bridge | 5 |
| Work Day | 5 |

**Locale scope:**
- National: 174
- Local: 152 (city-specific)
- Regional: 24 (state-specific)

**Important nuances:**
- `transferred=True` means the holiday was moved to another date — treat as normal day
- `Transfer` type rows indicate the actual celebration date
- `Bridge` = extra day added to extend holiday weekend
- `Work Day` = makeup day for a Bridge (Saturday worked)
- `Additional` = extra days around major holidays (e.g., Christmas season)

---

## transactions.csv (Supplementary)

- 83,488 rows: daily transaction counts per store
- **Only covers train period** (2013-01-01 to 2017-08-15) — NOT available for test
- Range: 5 to 8,359 transactions/day, Mean: 1,695
- Useful for understanding patterns but must be forecasted or dropped for test predictions

---

## Domain Notes (Competition Context)

1. **Payday effect:** Public sector wages paid on 15th and last day of month — expect sales spikes
2. **Earthquake:** Magnitude 7.8 on 2016-04-16 — relief donations disrupted supermarket sales for weeks
3. **Oil dependency:** Ecuador's economy is tightly coupled to oil prices — include as feature
4. **Seasonality:** Expect weekly (day-of-week), monthly (payday), and yearly (holidays/seasons) patterns
5. **Promotions:** `onpromotion` is available in both train and test — strong predictive signal

---

## Data Quality Summary

| Check | Status |
|---|---|
| Nulls in train | None |
| Nulls in test | None |
| Nulls in oil | 43 (need interpolation) |
| Negative sales | None |
| Dense grid (no missing combos) | Yes — 54 x 33 x days = exact row count |
| Test dates contiguous with train | Yes — train ends 08-15, test starts 08-16 |
| Transactions in test period | No — train-only feature |

---

## Modeling Considerations

- **Scale:** 3M train rows, 1,782 time series (54 stores x 33 families)
- **Forecast horizon:** 15 days ahead
- **Zero-inflated:** 31% zeros — consider separate zero/non-zero modeling or log1p transform
- **Hierarchical structure:** store > city > state; family groupings; cluster groupings
- **Available at test time:** date features, store metadata, oil prices, holidays, onpromotion
- **NOT available at test time:** transactions (must be predicted or excluded)
