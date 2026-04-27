## Overview

This competition is derived from **NOAA CO-OPS** water level observations. The task is to forecast **extreme low-water anomalies** over the next 12–24 hours, relative to a recent baseline (past 72-hour mean).

The label is station-relative (train-only 10th percentile thresholds) and the split is time-based with a regime shift, making the task calibration-sensitive.

## Source

- NOAA CO-OPS data API (water level): `https://api.tidesandcurrents.noaa.gov/api/prod/`

This competition reads cached CO-OPS JSON responses already present in this workspace.

## License

Public-Domain-US-Gov

NOAA is a U.S. government agency; CO-OPS observations are generally public domain.

## Features

All numeric signals are released as **train-fitted quantile bins**; `-1` means missing.

- **Panel identifiers**: `row_id`, `station_token`, `horizon_h`
- **Coarsened time & space**: `event_era`, `month`, `hour`, `dow`, `is_weekend`, `spatial_bin`
- **Slice**: `slice_high_tidal_range` (spring-tide proxy; threshold fit on training rows only)
- **Binned water-level features**:
  - level/changes: `wl_level_bin`, `wl_lag1_bin`, `wl_chg1_bin`
  - rolling summaries: `wl_mean_6h_bin`, `wl_std_24h_bin`, `wl_range_24h_bin`, `wl_mean_72h_bin`
  - data quality: `sigma_mean_6h_bin`, `missing_24h_bin`, `missing_count`
  - coarse spatial bins: `lat_bin_1p0_bin`, `lon_bin_1p0_bin`

## Splitting & Leakage

- **Split policy (deterministic, time-based)**:
  - Train: years \(\le 2018\)
  - Bridge: 2019–2020 (deterministic hashed assignment to test)
  - Test: years \(\ge 2021\)
- **Leakage mitigations**:
  - exact timestamps and station identifiers are excluded (only hashed ids and coarse time/space context)
  - bin edges are learned from training rows only
  - the label depends on future hours not available to features

