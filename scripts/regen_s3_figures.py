"""Regenerate only the 3 Section 3 figures from saved predictions parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids Windows handle issues

import pandas as pd
from src import config as C
from src.splits import ALL_ELIGIBLE_CHANNELS
from src.features import prepare_house_data
from src.evaluate import generate_s3_figures

test_house = 6
ch = ALL_ELIGIBLE_CHANNELS[test_house]

print("Loading test house data...")
_, y_test, df_test_raw = prepare_house_data(test_house, ch)

print("Loading saved predictions...")
pred_df = pd.read_parquet(C.OUT_DIR / "test_predictions.parquet")

# Align to y_test index
common_idx = y_test.index.intersection(pred_df.index)
y_true   = y_test.loc[common_idx]
s_lgbm   = pred_df.loc[common_idx, "pred_lgbm_w"]
s_rule   = pred_df.loc[common_idx, "pred_rule_w"]
s_b2     = pred_df.loc[common_idx, "pred_b2_w"]
s_b1     = pred_df.loc[common_idx, "pred_b1_w"]

on_threshold = 20.0  # confirmed from pipeline output

print(f"Generating figures for {len(common_idx):,} rows, threshold={on_threshold} W ...")
generate_s3_figures(
    test_house=test_house,
    df_raw=df_test_raw,
    y_true=y_true,
    y_pred_lgbm=s_lgbm,
    y_pred_rule=s_rule,
    y_pred_b1=s_b1,
    y_pred_b2=s_b2,
    on_threshold=on_threshold,
)
print("Done.")
