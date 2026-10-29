"""Washing machine cycle detection logic for REFIT NILM project."""

import numpy as np
import pandas as pd
from . import config as C


def long_runs(mask, min_len: int):
    """Return boolean array of runs of True in mask with length >= min_len."""
    m = np.asarray(mask, bool)
    out = np.zeros_like(m)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= min_len:
            out[s:e] = True
    return out


def run_lengths(mask):
    """Return array of run lengths for True in mask."""
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def mask_appliance_flatline(p: pd.Series, on_w: float = C.ON_W, stale_min: int = C.STALE_MIN) -> pd.Series:
    """Mask stretches where an individual appliance channel remains constant (<0.01 W diff) for >= stale_min while ON (> on_w).

    Physical justification: A real domestic appliance never draws constant non-zero active power
    down to 0.00 W standard deviation for 15+ minutes. Heating elements pulse via bimetallic thermostats
    and motors fluctuate during drum agitation. Constant non-zero power indicates sensor transmitter freeze.
    """
    p_out = p.copy()
    diff = p_out.diff().abs()
    flat = (diff < 0.01) & (p_out > on_w)
    m = np.asarray(flat, bool)
    stale = np.zeros_like(m)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        # K zero-diffs represent K + 1 identical consecutive values
        if (e - s) >= (stale_min - 1):
            stale[max(s - 1, 0):e] = True
    p_out[stale] = np.nan
    return p_out


def detect_cycles(
    p: pd.Series,
    on_w: float = C.ON_W,
    merge_gap: int = C.MERGE_GAP_MIN,
    min_dur: int = C.MIN_CYCLE_MIN,
    min_wh: float = C.MIN_CYCLE_WH,
    max_dur: int = C.MAX_CYCLE_MIN,
    max_wh: float = C.MAX_CYCLE_WH,
) -> pd.DataFrame:
    """Detect appliance operating cycles from 1-minute power time series.

    Parameters
    ----------
    p : pd.Series
        1-minute mean power series in Watts, with regular DatetimeIndex.
        NaN indicates unusable/missing minutes.
    on_w : float
        Threshold in Watts above which appliance is considered ON.
    merge_gap : int
        Maximum gap in minutes between ON segments to merge into a single cycle.
    min_dur : int
        Minimum duration in minutes for a valid cycle.
    min_wh : float
        Minimum active energy in Watt-hours for a valid cycle.
    max_dur : int
        Maximum duration in minutes for a valid cycle (physical ceiling; default 300 min).
    max_wh : float
        Maximum active energy in Watt-hours for a valid cycle (physical ceiling; default 3500 Wh).

    Returns
    -------
    pd.DataFrame
        Table of detected cycles with columns:
        [start, end, duration_min, energy_wh, peak_w, mean_w]
    """
    if p.empty:
        return pd.DataFrame(columns=["start", "end", "duration_min", "energy_wh", "peak_w", "mean_w"])

    on = (p > on_w).fillna(False).to_numpy()
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    runs = [[s, e] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

    merged = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    v = p.fillna(0).to_numpy()
    nan = p.isna().to_numpy()
    rows = []
    for s, e in merged:
        seg = v[s:e]
        wh = float(seg.sum() / 60.0)
        # Check boundary or internal NaN
        touches_gap = bool(nan[max(s - 1, 0)] or nan[min(e, len(nan) - 1)] or nan[s:e].any())
        dur = e - s
        if (min_dur <= dur <= max_dur) and (min_wh <= wh <= max_wh) and not touches_gap:
            rows.append({
                "start": p.index[s],
                "end": p.index[e - 1],
                "duration_min": int(dur),
                "energy_wh": float(wh),
                "peak_w": float(seg.max()),
                "mean_w": float(seg.mean()),
            })

    cols = ["start", "end", "duration_min", "energy_wh", "peak_w", "mean_w"]
    return pd.DataFrame(rows, columns=cols)
