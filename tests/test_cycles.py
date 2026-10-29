"""Tests for cycle detection logic."""

import pandas as pd
import numpy as np
import pytest

from src.cycles import detect_cycles


def test_constant_1000w_energy():
    """Energy for a constant 1000 W over 60 min is 1000 Wh."""
    idx = pd.date_range("2014-01-01 00:00:00", periods=80, freq="1min", tz="UTC")
    # Buffer with 10 min 0W before and after so it doesn't touch index boundaries
    p = pd.Series(0.0, index=idx)
    p.iloc[10:70] = 1000.0  # exactly 60 minutes of 1000 W

    cycles = detect_cycles(p)
    assert len(cycles) == 1
    c = cycles.iloc[0]
    assert c["duration_min"] == 60
    assert np.isclose(c["energy_wh"], 1000.0)
    assert np.isclose(c["peak_w"], 1000.0)
    assert np.isclose(c["mean_w"], 1000.0)


def test_merge_short_pause():
    """A pause shorter than the merge gap is merged into one cycle."""
    idx = pd.date_range("2014-01-01 00:00:00", periods=100, freq="1min", tz="UTC")
    p = pd.Series(0.0, index=idx)
    # Segment 1: 20 min @ 500 W
    p.iloc[10:30] = 500.0
    # Pause: 5 min @ 0 W (<= MERGE_GAP_MIN of 10 min)
    p.iloc[30:35] = 0.0
    # Segment 2: 25 min @ 500 W
    p.iloc[35:60] = 500.0

    cycles = detect_cycles(p, merge_gap=10)
    assert len(cycles) == 1
    c = cycles.iloc[0]
    # Total duration from index 10 to 60 is 50 minutes
    assert c["duration_min"] == 50
    # Energy: 45 active minutes @ 500 W = 45 * 500 / 60 = 375 Wh
    assert np.isclose(c["energy_wh"], 375.0)


def test_short_blip_dropped():
    """A short blip (< MIN_CYCLE_MIN or < MIN_CYCLE_WH) is dropped."""
    idx = pd.date_range("2014-01-01 00:00:00", periods=50, freq="1min", tz="UTC")
    p = pd.Series(0.0, index=idx)
    # 5 min blip @ 100 W (duration 5 < 10 min threshold)
    p.iloc[10:15] = 100.0

    cycles = detect_cycles(p, min_dur=10, min_wh=20.0)
    assert len(cycles) == 0


def test_cycle_touching_nan_discarded():
    """A cycle next to or touching a NaN is discarded."""
    idx = pd.date_range("2014-01-01 00:00:00", periods=60, freq="1min", tz="UTC")
    p = pd.Series(0.0, index=idx)
    p.iloc[10:40] = 500.0  # 30 min cycle
    # Put a NaN immediately before the cycle (at index 9)
    p.iloc[9] = np.nan

    cycles = detect_cycles(p)
    assert len(cycles) == 0

    # Test NaN inside the cycle
    p2 = pd.Series(0.0, index=idx)
    p2.iloc[10:40] = 500.0
    p2.iloc[25] = np.nan
    cycles2 = detect_cycles(p2)
    assert len(cycles2) == 0
