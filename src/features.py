"""Feature engineering for NILM disaggregation model (Phase 6).

Builds rolling, lagging, leading, and temporal features exclusively from whole-home
aggregate active power and timestamp. All sub-metered channels are excluded from inputs.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from . import config as C
from .cycles import long_runs, mask_appliance_flatline

OFFSETS = [-30, -20, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30]
ROLLING_WINDOWS = (5, 15, 31)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate-derived features from regular 1-minute time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regular 1-min DatetimeIndex (UTC) and 'aggregate' column.
        Unusable minutes must already be set to NaN so that rolling windows
        touching unusable data naturally evaluate to NaN and are dropped.
    """
    a = df["aggregate"]
    X = {f"agg_t{k:+d}": a.shift(-k) for k in OFFSETS}

    for w in ROLLING_WINDOWS:
        r = a.rolling(w, center=True, min_periods=w)
        X.update({
            f"mean_{w}": r.mean(),
            f"std_{w}": r.std(),
            f"min_{w}": r.min(),
            f"max_{w}": r.max(),
        })

    X["d_prev"] = a - a.shift(1)
    X["d_next"] = a.shift(-1) - a
    X["above_base60"] = a - a.rolling(61, center=True, min_periods=61).min()

    # Time features in Europe/London local time
    loc = a.index.tz_convert(C.TZ_LOCAL)
    h = loc.hour + loc.minute / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    X["dow_sin"] = np.sin(2 * np.pi * loc.day_of_week / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * loc.day_of_week / 7.0)

    features_df = pd.DataFrame(X, index=a.index)
    return features_df


def prepare_house_data(house: int, channel: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load house data, apply quality masks, build features and target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with NaNs dropped.
    y : pd.Series
        Target series (Washing machine active power in Watts, clipped at >= 0).
    raw_df : pd.DataFrame
        Original 1-minute DataFrame with masks and unclipped target for evaluation.
    """
    p_path = C.PROC_DIR / f"house_{house}_1min.parquet"
    df = pd.read_parquet(p_path)

    # 1. Compute quality masks
    channels = ["aggregate"] + [f"a{i}" for i in range(1, 10)]
    present = df["aggregate"].notna().to_numpy()
    diffs = df[channels].diff().abs()
    unchanged = (diffs < 0.01).all(axis=1) & present
    stale_all = long_runs(unchanged, C.STALE_MIN)
    bad_issue = (df["issues"] == 1).fillna(False).to_numpy()

    # Appliance-level sensor freeze
    wm_masked = mask_appliance_flatline(df[channel], on_w=C.ON_W, stale_min=C.STALE_MIN)
    app_stale = wm_masked.isna().to_numpy()

    usable = present & ~stale_all & ~bad_issue & ~app_stale

    # 2. Mask unusable minutes in both aggregate and target
    df_masked = df[["aggregate", channel]].copy()
    df_masked.loc[~usable, "aggregate"] = np.nan
    df_masked.loc[~usable, channel] = np.nan

    # 3. Build features on masked series
    X_full = build_features(df_masked)
    y_full = df_masked[channel].clip(lower=0.0)

    # 4. Drop any row where features or target are NaN (drops windows touching unusable minutes)
    valid_idx = X_full.notna().all(axis=1) & y_full.notna()
    X = X_full.loc[valid_idx]
    y = y_full.loc[valid_idx]

    # Store metadata in df for downstream evaluation
    df["usable"] = usable
    return X, y, df
