"""Cleaning and resampling utilities for REFIT data."""

from typing import List, Optional
import pandas as pd
import numpy as np


def resample_to_1min(df: pd.DataFrame, power_cols: Optional[List[str]] = None, issues_col: Optional[str] = "issues") -> pd.DataFrame:
    """Resample high-frequency time series to regular 1-minute bins.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with UTC-aware DatetimeIndex.
    power_cols : list of str, optional
        Names of power channels to resample using mean. If None, all columns except issues_col.
    issues_col : str, optional
        Name of the issues column to resample using max.

    Returns
    -------
    pd.DataFrame
        1-minute regularly sampled DataFrame.
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    if power_cols is None:
        power_cols = [c for c in df.columns if c != issues_col]

    resampler = df.resample("1min")
    power_resampled = resampler[power_cols].mean().astype("float32")

    if issues_col in df.columns:
        issues_resampled = resampler[issues_col].max().fillna(0).astype("int8")
        power_resampled[issues_col] = issues_resampled

    return power_resampled
