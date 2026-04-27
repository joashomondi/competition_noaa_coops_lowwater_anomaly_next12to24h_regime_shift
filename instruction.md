## Objective

Predict whether an anonymized NOAA CO-OPS tide-gauge station will experience an **unusually low water-level anomaly** within the next **12–24 hours**.

Each row is a **station-hour × horizon** example (`horizon_h ∈ {12, 24}`), derived from the CO-OPS `water_level` product (hourly aggregated). Your model must output a calibrated probability in \([0,1]\).

## Target definition (train only)

Let \(w_t\) be the station’s hourly mean water level (meters, MSL). Let \(b_t\) be the station’s **past-only** 72-hour mean baseline:

\[
b_t = \text{mean}(w_{t-1},\dots,w_{t-72})
\]

Define the future minimum water level over the next \(H\) hours:

\[
m_{t,H} = \min(w_{t+1},\dots,w_{t+H})
\]

Define the future low-water anomaly relative to the baseline:

\[
\Delta_{t,H} = m_{t,H} - b_t
\]

For each station and horizon, compute a station-relative threshold from **training rows only**:

\[
\tau_{\text{station},H} = Q_{0.10}(\Delta_{t,H} \mid \text{station, train})
\]

Stations with limited training history fall back to the global threshold \(Q_{0.10}(\Delta_{t,H}\mid \text{train})\).

The binary target is:

\[
\texttt{target\_lowwater\_anom\_next}=\mathbb{1}[\Delta_{t,H}\le \tau_{\text{station},H}]
\]

## Files

- `train.csv`: features + `target_lowwater_anom_next`
- `test.csv`: features only (no target)
- `solution.csv`: ground-truth for scoring (not for training)
- `sample_submission.csv`, `perfect_submission.csv`: format references

## Submission format

Create `submission.csv` with exactly:

- `row_id`
- `pred_lowwater_anom_next` (probability in **[0, 1]**)

## Metric (lower is better)

\[
\text{Score} = 0.60\cdot \text{LogLoss}_{all}
             + 0.25\cdot \text{LogLoss}_{\text{high-tidal-range slice}}
             + 0.15\cdot (1 - \text{AUPRC}_{all})
\]

Slice membership is provided as `slice_high_tidal_range` in both `train.csv` and `test.csv`.

Deterministic scoring command:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

## Regime shift & leakage notes

- Split is time-based under the hood (newer years emphasized in test), but exact timestamps are not exposed.
- Continuous values are released as **bins learned from training rows only**; missing is encoded as `-1` and summarized by `missing_count`.
- Avoid random CV; validate by `event_era` and by `station_token`.

