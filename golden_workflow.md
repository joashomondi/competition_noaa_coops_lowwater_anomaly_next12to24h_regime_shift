## What “low-water anomaly” means here (domain grounding)

- You are predicting **tail low-water drops** relative to a recent baseline:
  - baseline \(b_t\) is the past 72-hour mean (past-only)
  - the event is the future 12–24h minimum falling unusually far below that baseline
- This captures a mix of:
  - **astronomical tides** (spring/neap modulation)
  - **weather-driven setdown** (wind setup/setdown, pressure effects)
  - **measurement / data gaps** (which can create spurious minima if mishandled)

## Data checks (CO-OPS specific)

- **Tide-phase vs anomaly**
  - `wl_range_24h` is a strong proxy for spring vs neap tides (range high during spring tides).
  - The slice `slice_high_tidal_range` is upweighted; inspect its base rate and ensure it’s non-degenerate.
- **Quality proxies matter**
  - `sigma_mean_6h` and `missing_24h` correlate with data quality; low-water minima are sensitive to gaps.
  - Treat `*_bin == -1` as missing categories, not “low”.
- **Station heterogeneity**
  - Low-water anomaly distributions differ by station due to datum/local bathymetry and tidal regime.
  - Always check base rates by `station_token` and `event_era`.

## Validation strategy (avoid leakage)

Random CV overstates performance because the split is time-based and stations have persistent behavior.

Recommended:

- **Blocked time validation using `event_era`**
  - Train on older eras; validate on the newest era present in training.
  - Inspect calibration drift across eras (LogLoss-heavy metric).
- **Group sanity by `station_token`**
  - Ensure your validation includes unseen or later-era station behavior, not just memorized station base rates.
- **Slice-aware reporting**
  - Report metrics on:
    - all rows
    - `slice_high_tidal_range == 1`
    - each `horizon_h` (12h vs 24h)

## Baselines and tradeoffs

- **Regularized logistic regression**
  - One-hot encode binned features; keep `missing_count` numeric.
  - Often provides strong calibration, which matters for LogLoss.
- **GBDT**
  - Captures interactions like `station_token × hour × wl_range_24h_bin` (tide-phase effects).
  - Needs calibration; tree models can be overconfident on rare tail events.

High-signal interactions to test:

- `station_token × hour × wl_range_24h_bin`
- `event_era × wl_mean_72h_bin × wl_chg1_bin`
- `sigma_mean_6h_bin × missing_24h_bin` (spurious minima risk)

## Non-obvious failure modes

- **Spring-tide confounding**
  - Low-water minima are more extreme during spring tides; models that ignore `wl_range_24h` will be miscalibrated.
- **Gap-induced false lows**
  - Missingness can create artifacts in resampled series; overconfident predictions here explode LogLoss.
- **Station memorization**
  - Since thresholds are station-relative, a model can learn station base rates. This breaks under era drift.

## Calibration plan

- Evaluate reliability by `event_era` and `slice_high_tidal_range`.
- Calibrate on the newest training-era block (temperature scaling / Platt).
- If slice is miscalibrated, use a slice-conditional calibrator or conservative probability caps.

## Submission checklist

- `submission.csv` has exactly `row_id` and `pred_lowwater_anom_next`
- ids match `test.csv`
- predictions are finite and in [0,1]
- run:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

