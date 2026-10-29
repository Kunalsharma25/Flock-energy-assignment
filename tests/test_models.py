"""Tests for feature engineering and baseline models."""

import numpy as np
import pandas as pd
from src.features import build_features
from src.models import BaselineAlwaysOff, BaselineTimeOfDay, postprocess_predictions


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
