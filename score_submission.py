import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


ID_COLUMN = "row_id"
TARGET_COLUMN = "target_lowwater_anom_next"
PRED_COLUMN = "pred_lowwater_anom_next"
SLICE_COLUMN = "slice_high_tidal_range"


def _fail(msg: str) -> None:
    sys.stderr.write(str(msg).strip() + "\n")
    raise SystemExit(1)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        _fail(f"File not found: {path}")
    return pd.read_csv(path)


def _validate(sub: pd.DataFrame, sol: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for c in (ID_COLUMN, PRED_COLUMN):
        if c not in sub.columns:
            _fail(f"Submission missing required column: {c}")
    for c in (ID_COLUMN, TARGET_COLUMN):
        if c not in sol.columns:
            _fail(f"Solution missing required column: {c}")

    if sub[ID_COLUMN].isna().any() or sol[ID_COLUMN].isna().any():
        _fail("Missing row_id in submission or solution.")
    if sub[ID_COLUMN].duplicated().any() or sol[ID_COLUMN].duplicated().any():
        _fail("Duplicate row_id in submission or solution.")
    if len(sub) != len(sol):
        _fail(f"Row count mismatch: submission={len(sub)} solution={len(sol)}")

    sub_ids = pd.to_numeric(sub[ID_COLUMN], errors="coerce").to_numpy(dtype=np.int64)
    sol_ids = pd.to_numeric(sol[ID_COLUMN], errors="coerce").to_numpy(dtype=np.int64)
    if np.isnan(sub_ids).any() or np.isnan(sol_ids).any():  # type: ignore[arg-type]
        _fail("Non-numeric row_id detected.")

    sub_idx = np.argsort(sub_ids)
    sol_idx = np.argsort(sol_ids)
    if not np.array_equal(sub_ids[sub_idx], sol_ids[sol_idx]):
        _fail("Submission ids do not match solution ids.")

    sub_sorted = sub.iloc[sub_idx].reset_index(drop=True)
    sol_sorted = sol.iloc[sol_idx].reset_index(drop=True)

    y_true = pd.to_numeric(sol_sorted[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(sub_sorted[PRED_COLUMN], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y_true).any():
        _fail("NaN targets in solution.")
    if np.isnan(y_pred).any():
        _fail("NaN predictions in submission.")
    if ((y_true != 0) & (y_true != 1)).any():
        _fail("Targets must be binary 0/1.")
    if (y_pred < 0).any() or (y_pred > 1).any():
        _fail("Predictions must be in [0, 1].")

    if SLICE_COLUMN in sol_sorted.columns:
        sl = pd.to_numeric(sol_sorted[SLICE_COLUMN], errors="coerce").fillna(0).to_numpy(dtype=int)
        sl = (sl > 0).astype(int)
    else:
        sl = np.zeros_like(y_true, dtype=int)

    return y_true.astype(int), y_pred.astype(float), sl.astype(int)


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _average_precision(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision[y_sorted == 1]) / max(1, int(y.sum())))


def score(y_true: np.ndarray, y_pred: np.ndarray, sl: np.ndarray) -> float:
    ll_all = _log_loss(y_true, y_pred)
    ap_all = _average_precision(y_true, y_pred)
    mask = sl.astype(bool)
    ll_slice = _log_loss(y_true[mask], y_pred[mask]) if mask.any() else ll_all
    return float(0.60 * ll_all + 0.25 * ll_slice + 0.15 * (1.0 - ap_all))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--submission-path", type=Path, required=True)
    p.add_argument("--solution-path", type=Path, required=True)
    args = p.parse_args()

    sub = _read_csv(args.submission_path)
    sol = _read_csv(args.solution_path)
    y_true, y_pred, sl = _validate(sub, sol)
    s = score(y_true, y_pred, sl)
    if not np.isfinite(s):
        _fail("Non-finite score computed.")
    print(f"{float(s)}")


if __name__ == "__main__":
    main()

