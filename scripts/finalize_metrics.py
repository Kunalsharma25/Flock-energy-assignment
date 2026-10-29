"""Compute v2 primary test metrics (baselines + LightGBM) and produce clean final CSV."""
import sys, matplotlib
matplotlib.use("Agg")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src import config as C
from src.splits import ALL_ELIGIBLE_CHANNELS, PRIMARY_TRAIN
from src.features import prepare_house_data
from src.models import (BaselineAlwaysOff, BaselineTimeOfDay, BaselinePhysicalRule,
                         train_lightgbm, postprocess_predictions, zero_clamp,
                         optimize_on_threshold)
from src.evaluate import compute_metrics

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading train + val + test houses...")
train_houses = PRIMARY_TRAIN
val_house, test_house = 5, 6

house_data = {}
for h in set(train_houses + [val_house, test_house]):
    ch = ALL_ELIGIBLE_CHANNELS[h]
    house_data[h] = prepare_house_data(h, ch)
    print(f"  H{h}: {len(house_data[h][0]):,} rows")

X_train = pd.concat([house_data[h][0] for h in train_houses])
y_train = pd.concat([house_data[h][1] for h in train_houses])
X_val, y_val, df_val_raw   = house_data[val_house]
X_test, y_test, df_test_raw = house_data[test_house]

# ── Train model ───────────────────────────────────────────────────────────────
print(f"\nTraining LightGBM on {len(X_train):,} rows...")
model = train_lightgbm(X_train, y_train, X_val, y_val)
print(f"  Best iteration: {model.best_iteration_}")

# ── Threshold ─────────────────────────────────────────────────────────────────
val_raw   = model.predict(X_val)
val_preds = postprocess_predictions(val_raw, df_val_raw.loc[X_val.index, "aggregate"])
best_thresh, best_val_f1 = optimize_on_threshold(y_val, val_preds)
print(f"  Threshold: {best_thresh:.0f} W  Val-F1: {best_val_f1:.4f}")

# ── Primary test predictions ──────────────────────────────────────────────────
agg_test = df_test_raw.loc[X_test.index, "aggregate"]

lgbm_raw   = model.predict(X_test)
lgbm_preds = postprocess_predictions(lgbm_raw, agg_test)
lgbm_preds = zero_clamp(lgbm_preds, best_thresh)

b1 = BaselineAlwaysOff().fit(X_train, y_train)
b2 = BaselineTimeOfDay().fit(X_train, y_train)
b3 = BaselinePhysicalRule()

b1_preds  = b1.predict(X_test)
b2_preds  = b2.predict(X_test)
agg_full  = df_test_raw["aggregate"].reindex(y_test.index)
b3_preds  = b3.predict(agg_full)

s_true = y_test
s_lgbm = pd.Series(lgbm_preds, index=y_test.index)
s_b1   = pd.Series(b1_preds,   index=y_test.index)
s_b2   = pd.Series(b2_preds,   index=y_test.index)
s_b3   = pd.Series(b3_preds,   index=y_test.index)

m_lgbm = compute_metrics(s_true, s_lgbm, best_thresh, "LightGBM v2 (Primary Test)", test_house)
m_b3   = compute_metrics(s_true, s_b3,   best_thresh, "Baseline 3 (Physical Rule)", test_house)
m_b2   = compute_metrics(s_true, s_b2,   best_thresh, "Baseline 2 (Time-of-Day)",   test_house)
m_b1   = compute_metrics(s_true, s_b1,   best_thresh, "Baseline 1 (Always Off)",    test_house)

primary_rows = pd.DataFrame([m_lgbm, m_b3, m_b2, m_b1])

# ── Load LOHO rows (already saved) ───────────────────────────────────────────
existing = pd.read_csv(C.OUT_DIR / "model_metrics.csv")
loho_rows = existing[existing["model"].str.contains("LOHO LightGBM v2|Pooled", na=False)]

final_df = pd.concat([primary_rows, loho_rows], ignore_index=True)
out_path = C.OUT_DIR / "model_metrics.csv"
final_df.to_csv(out_path, index=False)

print(f"\nSaved {out_path}")
print("\n--- FINAL CLEAN METRICS ---")
print(final_df[["house","model","sae","f1","mae_all_w","cycle_recall","cycle_precision"]].to_string(index=False))
