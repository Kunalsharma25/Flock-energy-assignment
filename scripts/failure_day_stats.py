"""Compute reproducible statistics for the highest-MAE failure day (House 6).

Reads outputs/test_predictions.parquet and computes exact metrics for 2013-12-28,
saving the results to outputs/failure_day_stats.json.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT_PARQUET = ROOT / "outputs" / "test_predictions.parquet"
OUTPUT_JSON = ROOT / "outputs" / "failure_day_stats.json"


def main():
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PARQUET}")

    df = pd.read_parquet(INPUT_PARQUET)
    # Filter rows with known target
    df = df.dropna(subset=["true_wm_w"]).copy()

    # Localize time to Europe/London
    idx_local = df.index.tz_convert("Europe/London")
    df["local_date"] = idx_local.date
    df["local_time"] = idx_local.time

    # a) Highest daily MAE
    df["abs_err"] = (df["pred_lgbm_w"] - df["true_wm_w"]).abs()
    daily_mae = df.groupby("local_date")["abs_err"].mean()
    worst_day = daily_mae.idxmax()
    worst_mae = float(daily_mae.max())

    # Filter for 2013-12-28
    d28 = df[df["local_date"] == worst_day].copy()

    # b) 2013-12-28 totals
    true_wh = float(d28["true_wm_w"].sum() / 60.0)
    pred_wh = float(d28["pred_lgbm_w"].sum() / 60.0)
    true_peak_w = float(d28["true_wm_w"].max())

    # c) Heating minutes (true_wm_w > 1500)
    heating = d28[d28["true_wm_w"] > 1500]
    heat_count = int(len(heating))
    heat_med = float(heating["pred_lgbm_w"].median())
    heat_p10 = float(np.percentile(heating["pred_lgbm_w"], 10))
    heat_p90 = float(np.percentile(heating["pred_lgbm_w"], 90))
    heat_max = float(heating["pred_lgbm_w"].max())
    heat_true_wh = float(heating["true_wm_w"].sum() / 60.0)
    heat_pred_wh = float(heating["pred_lgbm_w"].sum() / 60.0)
    heat_share_ge_20 = float((heating["pred_lgbm_w"] >= 20).mean())

    # d) Low-power minutes (20 < true_wm_w < 400)
    low = d28[(d28["true_wm_w"] > 20) & (d28["true_wm_w"] < 400)]
    low_count = int(len(low))
    low_true_wh = float(low["true_wm_w"].sum() / 60.0)
    low_share_zero = float((low["pred_lgbm_w"] == 0).mean())

    # e) Window 11:40 - 12:05 local
    t_start = pd.to_datetime("11:40").time()
    t_end = pd.to_datetime("12:05").time()
    win = d28[(d28["local_time"] >= t_start) & (d28["local_time"] <= t_end)]
    win_count = int(len(win))
    win_max_true = float(win["true_wm_w"].max())
    win_max_pred = float(win["pred_lgbm_w"].max())
    win_fp_burst_count = int(((win["pred_lgbm_w"] >= 300) & (win["true_wm_w"] <= 150)).sum())
    win_max_agg = float(win["aggregate_w"].max())

    stats = {
        "highest_mae_day": str(worst_day),
        "highest_daily_mae": round(worst_mae, 2),
        "day_2013_12_28": {
            "true_wh": round(true_wh, 1),
            "pred_wh": round(pred_wh, 1),
            "true_peak_w": round(true_peak_w, 1),
        },
        "heating_minutes_gt_1500w": {
            "count": heat_count,
            "pred_median_w": round(heat_med, 1),
            "pred_p10_w": round(heat_p10, 1),
            "pred_p90_w": round(heat_p90, 1),
            "pred_max_w": round(heat_max, 1),
            "true_wh": round(heat_true_wh, 1),
            "pred_wh": round(heat_pred_wh, 1),
            "share_pred_ge_20w": round(heat_share_ge_20, 2),
        },
        "low_power_minutes_20_to_400w": {
            "count": low_count,
            "true_wh": round(low_true_wh, 1),
            "share_pred_zero": round(low_share_zero, 2),
        },
        "window_1140_to_1205": {
            "minute_count": win_count,
            "max_true_wm_w": round(win_max_true, 1),
            "max_pred_lgbm_w": round(win_max_pred, 1),
            "minutes_pred_ge_300_true_le_150": win_fp_burst_count,
            "max_aggregate_w": round(win_max_agg, 1),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats to {OUTPUT_JSON}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
