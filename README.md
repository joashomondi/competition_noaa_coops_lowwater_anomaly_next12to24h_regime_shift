## NOAA CO-OPS Low-Water Anomaly Risk (Next 12–24 Hours)

Kaggle-style competition package built from **NOAA CO-OPS** water level time series.

- **Task**: binary classification — predict `pred_lowwater_anomaly_next`
- **Target**: whether the next 12/24 hours contain an unusually low-water anomaly (panelized by horizon)
- **Split**: time-based regime shift (see `dataset_card.md`)
- **Metric**: composite LogLoss + slice LogLoss + (1 - AUPRC) (see `instruction.md`)

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

