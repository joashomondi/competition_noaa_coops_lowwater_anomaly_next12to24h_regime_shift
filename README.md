## NOAA CO-OPS Low-Water Anomaly Risk (Next 12–24 Hours)

Portfolio-ready, Kaggle-style prediction task built from **NOAA CO-OPS** water level time series.

### What you’re predicting
- **Task**: binary classification
- **Predict**: `pred_lowwater_anom_next`
- **Target**: `target_lowwater_anom_next` — a low-water anomaly occurs within the next **12 or 24 hours**
- **Panelization**: each station-hour becomes two examples via `horizon_h ∈ {12,24}`

### Data & evaluation highlights
- **Rows**: train **180,082**, test **200,008**
- **Positive rate (train)**: ~**0.100**
- **Split**: time-based regime shift (details in `dataset_card.md`)
- **Metric**: composite scorer (LogLoss overall + slice LogLoss + (1 − AUPRC)); see `instruction.md`
- **Slice**: `slice_high_tidal_range` emphasizes spring-tide conditions where false positives are common

### Repository contents
- `train.csv`, `test.csv`, `solution.csv`
- `sample_submission.csv`, `perfect_submission.csv`
- `build_dataset.py`, `build_meta.json`
- `score_submission.py`
- `instruction.md`, `golden_workflow.md`, `dataset_card.md`

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

Baseline tips:
- Treat `station_token` and `horizon_h` as categorical
- Calibrate probabilities (LogLoss-heavy metric)

