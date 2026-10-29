"""NILM disaggregation model architectures and baselines (Phase 6).

Implements:
  1. Baseline 1: Always Off (predicts 0 W)
  2. Baseline 2: Historical Mean by (weekday, hour) in local time
  3. Main Model: LightGBM regressor with early stopping and post-processing
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

from . import config as C


class BaselineAlwaysOff:
    """Baseline 1: Always predicts zero power."""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float32)


class BaselineTimeOfDay:
    """Baseline 2: Mean appliance power by (weekday, hour) in Europe/London local time."""
    def __init__(self):
        self.lookup_ = {}
        self.global_mean_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        loc = X.index.tz_convert(C.TZ_LOCAL)
        df_temp = pd.DataFrame({
            "dow": loc.day_of_week,
            "hour": loc.hour,
            "y": y.values,
        })
        self.lookup_ = df_temp.groupby(["dow", "hour"])["y"].mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        loc = X.index.tz_convert(C.TZ_LOCAL)
        keys = list(zip(loc.day_of_week, loc.hour))
        preds = np.array([self.lookup_.get(k, self.global_mean_) for k in keys], dtype=np.float32)
        return preds


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = C.SEED,
) -> lgb.LGBMRegressor:
    """Train LightGBM regressor with early stopping on validation house."""
    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=2000,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=callbacks,
    )
    return model


def postprocess_predictions(raw_preds: np.ndarray, agg_series: pd.Series) -> np.ndarray:
    """Post-process predictions: clip to [0, aggregate].

    A washing machine cannot draw negative power and cannot exceed whole-house aggregate.
    """
    clipped = np.clip(raw_preds, 0.0, agg_series.values)
    return clipped


def optimize_on_threshold(
    y_val_true: pd.Series,
    val_preds: np.ndarray,
    candidate_thresholds: list = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0],
    true_on_w: float = C.ON_W,
) -> Tuple[float, float]:
    """Find ON-state decision threshold that maximizes F1 on validation house."""
    true_on = (y_val_true.values > true_on_w).astype(int)
    best_thresh = candidate_thresholds[0]
    best_f1 = -1.0

    for t in candidate_thresholds:
        pred_on = (val_preds > t).astype(int)
        score = f1_score(true_on, pred_on, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    return float(best_thresh), float(best_f1)
