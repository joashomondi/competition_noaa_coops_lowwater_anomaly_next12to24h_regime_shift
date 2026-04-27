from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


COMP_DIR = Path(__file__).resolve().parent
ROOT = COMP_DIR.parent

UPSTREAM_CACHE_DIR = ROOT / "competition_noaa_coops_highwater_next48h_regime_shift" / "_cache"

ID_COLUMN = "row_id"
STATION_COLUMN = "station_token"
HORIZON_COLUMN = "horizon_h"
ERA_COLUMN = "event_era"
TARGET_COLUMN = "target_lowwater_anom_next"
PRED_COLUMN = "pred_lowwater_anom_next"
SLICE_COLUMN = "slice_high_tidal_range"
MISSING_COLUMN = "missing_count"

N_BINS = 16
MIN_TRAIN_ROWS = 12_000


@dataclass(frozen=True)
class Config:
    # Horizons (hours).
    horizons_h: Tuple[int, ...] = (12, 24)
    tail_quantile: float = 0.10  # negative tail threshold on delta_low (train only)
    min_train_hours_per_station: int = 1200

    # Past-only windows (hours).
    w_short: int = 6
    w_mid: int = 24
    w_long: int = 72

    # Regime-shift split.
    train_max_year: int = 2018
    bridge_years: Tuple[int, ...] = (2019, 2020)
    test_min_year: int = 2021
    bridge_test_rate: int = 35

    # Deterministic sampling (keeps size manageable).
    keep_percent_train: int = 10
    keep_percent_test: int = 18

    # Cap stations to bound runtime.
    max_stations: int = 14


def _hash_percent(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % 100


def _station_token(station_id: str) -> str:
    return hashlib.sha256(("coops:" + station_id).encode("utf-8")).hexdigest()[:12]


def _row_id(token: str, horizon_h: int, ts: pd.Timestamp) -> int:
    s = f"lw{horizon_h}_{token}_{ts:%Y-%m-%dT%H}"
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:15], 16)


def _event_era(year: int) -> int:
    if year <= 2012:
        return 0
    if year <= 2016:
        return 1
    if year <= 2020:
        return 2
    return 3


def _nanquantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([-1.0, 1.0], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.nanquantile(x, qs)
    edges = np.unique(edges.astype(float))
    if edges.size < 2:
        v = float(edges[0]) if edges.size else 0.0
        return np.array([v - 1.0, v + 1.0], dtype=float)
    return edges


def _bin_with_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape[0], -1, dtype=np.int16)
    mask = np.isfinite(arr)
    if mask.any():
        idx = np.digitize(arr[mask], edges, right=False) - 1
        idx = np.clip(idx, 0, max(0, edges.size - 2))
        out[mask] = idx.astype(np.int16)
    return pd.Series(out, index=series.index, dtype="Int16")


def _read_cache_json(path: Path) -> Tuple[dict, List[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if "error" in obj:
        return {}, []
    meta = obj.get("metadata") or {}
    data = obj.get("data") or []
    if not isinstance(meta, dict) or not isinstance(data, list):
        return {}, []
    return meta, data


def _load_station_hourly(cfg: Config) -> pd.DataFrame:
    if not UPSTREAM_CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing upstream cache dir: {UPSTREAM_CACHE_DIR}")

    files = sorted(UPSTREAM_CACHE_DIR.glob("*.json"))
    if not files:
        raise RuntimeError("No cached CO-OPS .json files found.")

    station_files: Dict[str, List[Path]] = {}
    for p in files:
        meta, data = _read_cache_json(p)
        if not meta or not data:
            continue
        sid = str(meta.get("id", "")).strip()
        if not sid:
            continue
        station_files.setdefault(sid, []).append(p)

    if not station_files:
        raise RuntimeError("No usable cached station files found.")

    sids = sorted(
        station_files.keys(),
        key=lambda s: int(hashlib.md5(("sid:" + s).encode("utf-8")).hexdigest()[:8], 16),
    )[: cfg.max_stations]

    frames = []
    for sid in sids:
        parts = []
        lat = None
        lon = None
        for p in station_files[sid]:
            meta, rows = _read_cache_json(p)
            if not rows:
                continue
            if lat is None:
                try:
                    lat = float(meta.get("lat"))
                except Exception:
                    lat = None
            if lon is None:
                try:
                    lon = float(meta.get("lon"))
                except Exception:
                    lon = None

            df = pd.DataFrame(rows)
            if "t" not in df.columns or "v" not in df.columns:
                continue
            df["ts"] = pd.to_datetime(df["t"], errors="coerce", utc=False)
            df["wl_m"] = pd.to_numeric(df["v"], errors="coerce")
            df["sigma_m"] = pd.to_numeric(df.get("s"), errors="coerce")
            df = df[df["ts"].notna()].copy()
            df = df[np.isfinite(df["wl_m"].to_numpy(dtype=float))].copy()
            if df.empty:
                continue
            parts.append(df[["ts", "wl_m", "sigma_m"]])

        if not parts:
            continue
        raw = pd.concat(parts, axis=0, ignore_index=True)
        raw = raw.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        raw = raw.set_index("ts").sort_index()

        hourly = raw.resample("1h").agg(wl_m=("wl_m", "mean"), sigma_m=("sigma_m", "mean"))
        hourly = hourly.reset_index()
        hourly["station_id_raw"] = sid
        hourly["lat"] = lat
        hourly["lon"] = lon
        frames.append(hourly)

    if not frames:
        raise RuntimeError("No hourly data built from cache.")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out[out["ts"].notna()].copy()
    out = out.sort_values(["station_id_raw", "ts"]).reset_index(drop=True)
    return out


def _engineer_station(sub: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sub = sub.sort_values("ts").reset_index(drop=True)
    wl = pd.to_numeric(sub["wl_m"], errors="coerce")
    sig = pd.to_numeric(sub["sigma_m"], errors="coerce")

    wl_lag1 = wl.shift(1)
    sub["wl_level"] = wl
    sub["wl_lag1"] = wl_lag1
    sub["wl_chg1"] = wl_lag1 - wl.shift(2)

    sub["wl_mean_6h"] = wl_lag1.rolling(cfg.w_short, min_periods=3).mean()
    sub["wl_std_24h"] = wl_lag1.rolling(cfg.w_mid, min_periods=12).std()
    sub["wl_range_24h"] = wl_lag1.rolling(cfg.w_mid, min_periods=12).max() - wl_lag1.rolling(cfg.w_mid, min_periods=12).min()
    sub["wl_mean_72h"] = wl_lag1.rolling(cfg.w_long, min_periods=24).mean()
    sub["sigma_mean_6h"] = sig.shift(1).rolling(cfg.w_short, min_periods=3).mean()
    sub["missing_24h"] = wl_lag1.isna().astype(int).rolling(cfg.w_mid, min_periods=1).sum().astype(float)

    return sub


def main() -> None:
    cfg = Config()
    hourly = _load_station_hourly(cfg)

    hourly[STATION_COLUMN] = hourly["station_id_raw"].map(lambda s: _station_token(str(s))).astype(str)
    hourly["lat_bin_1p0"] = pd.to_numeric(hourly["lat"], errors="coerce").round(0)
    hourly["lon_bin_1p0"] = pd.to_numeric(hourly["lon"], errors="coerce").round(0)
    hourly["spatial_bin"] = (
        hourly["lat_bin_1p0"].astype("Int64").astype(str) + "_" + hourly["lon_bin_1p0"].astype("Int64").astype(str)
    )

    hourly["year"] = hourly["ts"].dt.year.astype(int)
    hourly["month"] = hourly["ts"].dt.month.astype(int)
    hourly["dayofyear"] = hourly["ts"].dt.dayofyear.astype(int)
    hourly["hour"] = hourly["ts"].dt.hour.astype(int)
    hourly["dow"] = hourly["ts"].dt.dayofweek.astype(int)
    hourly["is_weekend"] = (hourly["dow"] >= 5).astype(int)
    hourly[ERA_COLUMN] = hourly["year"].map(_event_era).astype(int)

    # Engineer features per station.
    feats = []
    for _, sub in hourly.groupby(STATION_COLUMN, sort=False):
        sub2 = _engineer_station(sub, cfg)
        feats.append(sub2)
    df = pd.concat(feats, axis=0, ignore_index=True)

    # Split assignment.
    years = df["year"].astype(int)
    is_bridge = years.isin(cfg.bridge_years)
    key = df[STATION_COLUMN].astype(str) + "_" + df["dayofyear"].astype(int).astype(str) + "_" + df["hour"].astype(int).astype(str)
    key_pct = key.map(_hash_percent).astype(int)
    is_test = years >= cfg.test_min_year
    is_test |= is_bridge & (key_pct < cfg.bridge_test_rate)
    is_test &= years > cfg.train_max_year

    # Deterministic sampling to control file size.
    keep_key = df[STATION_COLUMN].astype(str) + "|" + df["ts"].dt.strftime("%Y-%m-%dT%H")
    keep_pct = keep_key.map(_hash_percent).astype(int)
    keep = (~is_test & (keep_pct < cfg.keep_percent_train)) | (is_test & (keep_pct < cfg.keep_percent_test))
    df = df[keep].reset_index(drop=True)
    is_test = is_test[keep].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows after sampling.")

    # Create panel across horizons.
    panel_rows = []
    taus: Dict[str, Dict[str, float]] = {}

    for h in cfg.horizons_h:
        sub = df.copy()
        sub[HORIZON_COLUMN] = int(h)

        frames2 = []
        for st, ss in sub.groupby(STATION_COLUMN, sort=False):
            ss = ss.sort_values("ts").reset_index(drop=True)
            wl = pd.to_numeric(ss["wl_level"], errors="coerce")
            fut_min = wl.shift(-1).rolling(int(h), min_periods=max(2, int(h) // 2)).min().shift(-(int(h) - 1))
            ss["fut_min_wl"] = fut_min
            ss["delta_low"] = ss["fut_min_wl"] - ss["wl_mean_72h"]  # relative to recent baseline (past-only)
            frames2.append(ss)
        sub = pd.concat(frames2, axis=0, ignore_index=True)

        sub = sub[np.isfinite(sub["delta_low"].to_numpy(dtype=float))].copy()
        sub = sub[sub["wl_mean_72h"].notna()].copy()
        if sub.empty:
            continue

        # Train split for thresholding.
        train_sub = sub[~is_test.loc[sub.index]].copy()
        if train_sub.empty:
            continue

        # Station-specific low-tail thresholds (train only), with global fallback.
        global_tau = float(np.nanquantile(train_sub["delta_low"].to_numpy(dtype=float), cfg.tail_quantile))
        thr_by_station: Dict[str, float] = {}
        counts = train_sub.groupby(STATION_COLUMN)["delta_low"].count().to_dict()
        for st, gg in train_sub.groupby(STATION_COLUMN):
            if int(counts.get(str(st), 0)) >= cfg.min_train_hours_per_station:
                v = gg["delta_low"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    thr_by_station[str(st)] = float(np.quantile(v, cfg.tail_quantile))

        sub["tau_station"] = sub[STATION_COLUMN].map(lambda s: float(thr_by_station.get(str(s), global_tau))).astype(float)
        sub[TARGET_COLUMN] = (sub["delta_low"].to_numpy(dtype=float) <= sub["tau_station"].to_numpy(dtype=float)).astype(int)

        taus[f"h{h}"] = {"global_tau_q10": global_tau, "n_station_specific": int(len(thr_by_station))}
        panel_rows.append(sub)

    if not panel_rows:
        raise RuntimeError("No panel rows produced.")

    panel = pd.concat(panel_rows, axis=0, ignore_index=True)

    # Recompute is_test for panel (based on time fields).
    p_year = panel["year"].astype(int)
    p_is_bridge = p_year.isin(cfg.bridge_years)
    p_key = panel[STATION_COLUMN].astype(str) + "_" + panel["dayofyear"].astype(int).astype(str) + "_" + panel["hour"].astype(int).astype(str)
    p_key_pct = p_key.map(_hash_percent).astype(int)
    panel_is_test = p_year >= cfg.test_min_year
    panel_is_test |= p_is_bridge & (p_key_pct < cfg.bridge_test_rate)
    panel_is_test &= p_year > cfg.train_max_year

    train_panel = panel[~panel_is_test].copy()
    if train_panel.empty:
        raise RuntimeError("Empty panel train split.")

    # Slice: high tidal range (spring-tide proxy), threshold fit on training rows only.
    tr_vals = train_panel["wl_range_24h"].to_numpy(dtype=float)
    tr_vals = tr_vals[np.isfinite(tr_vals)]
    tr_thr = float(np.quantile(tr_vals, 0.75)) if tr_vals.size else float("inf")
    panel[SLICE_COLUMN] = (panel["wl_range_24h"].to_numpy(dtype=float) >= tr_thr).astype(int)

    # Train-only bin edges.
    to_bin = [
        "wl_level",
        "wl_lag1",
        "wl_chg1",
        "wl_mean_6h",
        "wl_std_24h",
        "wl_range_24h",
        "wl_mean_72h",
        "sigma_mean_6h",
        "missing_24h",
        "lat_bin_1p0",
        "lon_bin_1p0",
    ]

    edges: Dict[str, list] = {}
    for c in to_bin:
        e = _nanquantile_edges(train_panel[c].to_numpy(dtype=float), N_BINS)
        edges[c] = e.tolist()
        panel[f"{c}_bin"] = _bin_with_edges(panel[c], e)

    panel[MISSING_COLUMN] = panel[to_bin].isna().sum(axis=1).astype(int)

    # Stable row_id and deterministic permutation.
    panel[ID_COLUMN] = panel.apply(
        lambda r: _row_id(str(r[STATION_COLUMN]), int(r[HORIZON_COLUMN]), pd.Timestamp(r["ts"])),
        axis=1,
    ).astype("int64")

    perm_key = (
        panel[STATION_COLUMN].astype(str)
        + "|"
        + panel["ts"].dt.strftime("%Y-%m-%dT%H")
        + "|"
        + panel[HORIZON_COLUMN].astype(str)
    ).map(lambda s: int(hashlib.md5(("perm:" + s).encode("utf-8")).hexdigest()[:8], 16))
    order = np.argsort(perm_key.to_numpy(dtype=np.int64), kind="mergesort")
    panel = panel.iloc[order].reset_index(drop=True)
    panel_is_test = panel_is_test.iloc[order].reset_index(drop=True)

    out_cols = [
        ID_COLUMN,
        STATION_COLUMN,
        HORIZON_COLUMN,
        ERA_COLUMN,
        "spatial_bin",
        "month",
        "hour",
        "dow",
        "is_weekend",
        SLICE_COLUMN,
        MISSING_COLUMN,
    ] + [f"{c}_bin" for c in to_bin]

    out = panel[out_cols + [TARGET_COLUMN]].copy()
    train_out = out[~panel_is_test].copy()
    test_out = out[panel_is_test].copy()
    if train_out.empty or test_out.empty:
        raise RuntimeError("Build produced empty train or test split.")
    if len(train_out) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Train set too small ({len(train_out)} rows). Need >= {MIN_TRAIN_ROWS}.")

    train_out.to_csv(COMP_DIR / "train.csv", index=False)
    test_out[out_cols].to_csv(COMP_DIR / "test.csv", index=False)
    test_out[[ID_COLUMN, TARGET_COLUMN, SLICE_COLUMN]].to_csv(COMP_DIR / "solution.csv", index=False)

    sample = pd.DataFrame({ID_COLUMN: test_out[ID_COLUMN].to_numpy(), PRED_COLUMN: 0.5})
    sample.to_csv(COMP_DIR / "sample_submission.csv", index=False)

    y = test_out[TARGET_COLUMN].astype(int).to_numpy()
    perf_p = np.where(y == 1, 0.999, 0.001)
    perfect = pd.DataFrame({ID_COLUMN: test_out[ID_COLUMN].to_numpy(), PRED_COLUMN: perf_p})
    perfect.to_csv(COMP_DIR / "perfect_submission.csv", index=False)

    meta = {
        "source_cache_dir": str(UPSTREAM_CACHE_DIR),
        "horizons_h": list(cfg.horizons_h),
        "tail_quantile": float(cfg.tail_quantile),
        "min_train_hours_per_station": int(cfg.min_train_hours_per_station),
        "thresholds_summary": taus,
        "slice_threshold_train_only": {"wl_range_24h_q75": tr_thr},
        "split": {
            "train_max_year": int(cfg.train_max_year),
            "bridge_years": list(cfg.bridge_years),
            "test_min_year": int(cfg.test_min_year),
            "bridge_test_rate": int(cfg.bridge_test_rate),
            "keep_percent_train": int(cfg.keep_percent_train),
            "keep_percent_test": int(cfg.keep_percent_test),
        },
        "n_bins": int(N_BINS),
        "bin_edges": edges,
        "row_counts": {"train": int(len(train_out)), "test": int(len(test_out))},
        "positive_rate": {"train": float(train_out[TARGET_COLUMN].mean()), "test": float(test_out[TARGET_COLUMN].mean())},
        "stations_used": int(panel[STATION_COLUMN].nunique()),
    }
    (COMP_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print("Wrote competition files to", COMP_DIR)
    print("train rows:", int(len(train_out)), "test rows:", int(len(test_out)))
    print("train positive rate:", float(train_out[TARGET_COLUMN].mean()))
    print("test positive rate:", float(test_out[TARGET_COLUMN].mean()))


if __name__ == "__main__":
    main()

