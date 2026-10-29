"""Tests for resampling logic."""

import pandas as pd
import numpy as np
import pytest

from src.cleaning import resample_to_1min


def test_resample_constant_power():
    """A constant 100 W input resamples to 100 W, never 100 x samples."""
    # 8-second intervals over 5 minutes (approx 38 samples)
    idx = pd.date_range("2014-01-01 00:00:00", "2014-01-01 00:04:56", freq="8s", tz="UTC")
    df = pd.DataFrame({
        "aggregate": 100.0,
        "a1": 100.0,
        "issues": 0,
    }, index=idx)

    resampled = resample_to_1min(df)

    assert len(resampled) == 5
    assert np.allclose(resampled["aggregate"].dropna(), 100.0)
    assert np.allclose(resampled["a1"].dropna(), 100.0)
    # Ensure it's not the sum (which would be ~7-8 x 100 = 700-800 W)
    assert (resampled["aggregate"] < 200.0).all()
    assert (resampled["issues"] == 0).all()


def test_resample_issues_max():
    """Issues column takes max (1 if any sample has issues=1 in that minute)."""
    idx = pd.date_range("2014-01-01 00:00:00", "2014-01-01 00:01:52", freq="8s", tz="UTC")
    issues = np.zeros(len(idx), dtype=int)
    # Set one sample in minute 0 to 1
    issues[3] = 1

    df = pd.DataFrame({
        "aggregate": 250.0,
        "issues": issues,
    }, index=idx)

    resampled = resample_to_1min(df)
    assert resampled.loc["2014-01-01 00:00:00", "issues"] == 1
    assert resampled.loc["2014-01-01 00:01:00", "issues"] == 0
