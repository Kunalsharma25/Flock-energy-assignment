"""Tests for feature engineering and baseline models."""

import numpy as np
import pandas as pd
from src.features import build_features
from src.models import (
    BaselineAlwaysOff,
    BaselineTimeOfDay,
    BaselinePhysicalRule,
    postprocess_predictions,
    zero_clamp,
)


def test_build_features_shape_and_leakage():
    """Verify feature matrix columns and future offset window."""
    idx = pd.date_range("2014-01-01 00:00:00", periods=100, freq="1min", tz="UTC")
    df = pd.DataFrame({"aggregate": np.linspace(100, 500, 100)}, index=idx)
    X = build_features(df)
    assert "agg_t-30" in X.columns
    assert "agg_t+30" in X.columns
    assert "mean_5" in X.columns
    assert "hour_sin" in X.columns
    # Rolling windows and offsets produce NaNs at boundaries
    assert X.iloc[0].isna().any()
    assert X.iloc[-1].isna().any()


def test_baseline_always_off():
    """Baseline 1 must always predict zero."""
    idx = pd.date_range("2014-01-01 00:00:00", periods=20, freq="1min", tz="UTC")
    X = pd.DataFrame(index=idx)
    b1 = BaselineAlwaysOff()
    preds = b1.predict(X)
    assert len(preds) == 20
    assert (preds == 0.0).all()


def test_postprocessing_clipping():
    """Predictions must never be negative and never exceed concurrent aggregate load."""
    agg = pd.Series([100.0, 500.0, 1000.0])
    raw_preds = np.array([-50.0, 750.0, 800.0])
    clipped = postprocess_predictions(raw_preds, agg)
    assert clipped[0] == 0.0  # clipped lower bound
    assert clipped[1] == 500.0  # clipped upper bound
    assert clipped[2] == 800.0  # within bounds


def test_zero_clamp():
    """Predictions below threshold must be clamped strictly to 0.0 W."""
    preds = np.array([0.0, 5.0, 9.9, 10.0, 25.0])
    clamped = zero_clamp(preds, threshold=10.0)
    assert np.array_equal(clamped, np.array([0.0, 0.0, 0.0, 10.0, 25.0]))


def test_baseline_physical_rule():
    """Rule baseline triggers when aggregate exceeds 60-min min by >= 1500W for >= 10 min."""
    idx = pd.date_range("2014-01-01 00:00:00", periods=50, freq="1min", tz="UTC")
    # Base power 200W, then a 15-minute 2000W pulse (lift of 1800W >= 1500W)
    agg_vals = [200.0] * 15 + [2000.0] * 15 + [200.0] * 20
    agg = pd.Series(agg_vals, index=idx)
    rule = BaselinePhysicalRule()
    preds = rule.predict(agg)
    # The 15-minute pulse should trigger the rule
    assert (preds[15:30] > 0.0).all()
    # Baseline periods before and after should be zero
    assert (preds[0:15] == 0.0).all()
    assert (preds[30:50] == 0.0).all()
