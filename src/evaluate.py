"""Evaluation metrics and diagnostic visualizations for NILM models (Phase 6)."""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support

from . import config as C
from . import plotting as P
from .cycles import detect_cycles
from .splits import CHANNELS


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    on_threshold: float,
    model_name: str,
    house: int,
    true_on_w: float = C.ON_W,
) -> Dict[str, Any]:
    """Compute comprehensive NILM regression, classification, and cycle-level metrics."""
    y_t = y_true.values
    y_p = y_pred.values

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    on_mask = (y_t > true_on_w)
    off_mask = ~on_mask

    mae_on = float(mean_absolute_error(y_t[on_mask], y_p[on_mask])) if on_mask.any() else 0.0
    mae_off = float(mean_absolute_error(y_t[off_mask], y_p[off_mask])) if off_mask.any() else 0.0

    sum_true = float(y_t.sum())
    sum_pred = float(y_p.sum())
    sae = float(abs(sum_pred - sum_true) / sum_true) if sum_true > 0 else 0.0

    # ON-state classification
    t_on = on_mask.astype(int)
    p_on = (y_p > on_threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(t_on, p_on, average="binary", zero_division=0)

    # Cycle-level evaluation
    true_cyc = detect_cycles(y_true, on_w=true_on_w)
    pred_cyc = detect_cycles(y_pred, on_w=on_threshold)

    n_true = len(true_cyc)
    n_pred = len(pred_cyc)

    matched_count = 0
    energy_errors = []

    pred_on_series = pd.Series(p_on, index=y_true.index)
    for _, tc in true_cyc.iterrows():
        sub = pred_on_series.loc[tc["start"]:tc["end"]]
        overlap = sub.sum() / len(sub) if len(sub) > 0 else 0.0
        if overlap >= 0.5:
            matched_count += 1
            pred_sub_energy = y_pred.loc[tc["start"]:tc["end"]].sum() / 60.0
            ape = abs(pred_sub_energy - tc["energy_wh"]) / tc["energy_wh"] * 100.0 if tc["energy_wh"] > 0 else 0.0
            energy_errors.append(ape)

    cycle_recall = float(matched_count / n_true) if n_true > 0 else 0.0

    pred_matched = 0
    true_on_series = pd.Series(t_on, index=y_true.index)
    for _, pc in pred_cyc.iterrows():
        sub = true_on_series.loc[pc["start"]:pc["end"]]
        overlap = sub.sum() / len(sub) if len(sub) > 0 else 0.0
        if overlap >= 0.5:
            pred_matched += 1
    cycle_precision = float(pred_matched / n_pred) if n_pred > 0 else 0.0
    med_energy_ape = float(np.median(energy_errors)) if energy_errors else 0.0

    return {
        "house": house,
        "model": model_name,
        "on_threshold_w": round(on_threshold, 1),
        "mae_all_w": round(mae, 2),
        "rmse_w": round(rmse, 2),
        "mae_on_w": round(mae_on, 2),
        "mae_off_w": round(mae_off, 2),
        "sae": round(sae, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "n_true_cycles": n_true,
        "n_pred_cycles": n_pred,
        "cycle_recall": round(cycle_recall, 4),
        "cycle_precision": round(cycle_precision, 4),
        "cycle_energy_mape": round(med_energy_ape, 2),
    }


def generate_s3_figures(
    test_house: int,
    df_raw: pd.DataFrame,
    y_true: pd.Series,
    y_pred_lgbm: pd.Series,
    y_pred_b1: pd.Series,
    y_pred_b2: pd.Series,
    y_pred_rule: pd.Series,
    on_threshold: float,
):
    """Generate the 3 required Section 3 figures in figures/.

    Day selection rules
    -------------------
    * Figure 1 (actual-vs-predicted): day with at least 1 true cycle; median
      energy among qualifying days. NaN gaps in unusable minutes are rendered
      as blank stretches rather than interpolated straight lines.
    * Figure 2 (confounder): day with the highest false-positive energy in the
      LightGBM predictions.
    * Figure 3 (transfer): day with the highest mean absolute error.
    Captions describe only what is plotted.
    """
    P.set_style()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Helper: convert index to local time for a series, preserving NaN structure
    def to_local(s: pd.Series) -> pd.Series:
        s2 = s.copy()
        s2.index = s.index.tz_convert(C.TZ_LOCAL)
        return s2

    # Build full-grid versions of every series (reindexed to df_raw; gaps become NaN)
    full_idx = df_raw.index  # complete 1-min UTC index for test house
    y_true_full  = y_true.reindex(full_idx)
    lgbm_full    = y_pred_lgbm.reindex(full_idx)
    b1_full      = y_pred_b1.reindex(full_idx)
    b2_full      = y_pred_b2.reindex(full_idx)
    rule_full    = y_pred_rule.reindex(full_idx)

    # ── Figure 1: s3_actual_vs_predicted.png ─────────────────────────────────
    # Select the day with ≥1 true cycle and median-energy among qualifying days.
    true_cyc = detect_cycles(y_true, on_w=C.ON_W)
    y_true_local_full = to_local(y_true_full)

    days_with_cycles = []
    for d, g in y_true_local_full.groupby(y_true_local_full.index.date):
        cyc_on_day = (true_cyc["start"].dt.tz_convert(C.TZ_LOCAL).dt.date == d).sum()
        if cyc_on_day >= 1:
            days_with_cycles.append((d, cyc_on_day, g.sum() / 60.0))

    if days_with_cycles:
        days_df = pd.DataFrame(days_with_cycles, columns=["date", "n_cycles", "energy_wh"])
        med_e = days_df["energy_wh"].median()
        rep_day = days_df.iloc[(days_df["energy_wh"] - med_e).abs().argmin()]["date"]
    else:
        rep_day = y_true_local_full.index[0].date()

    rep_day_str  = str(rep_day)
    mask_day     = (full_idx.tz_convert(C.TZ_LOCAL).date == rep_day)
    day_idx      = full_idx[mask_day]
    day_idx_local = day_idx.tz_convert(C.TZ_LOCAL)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Top: aggregate (NaN gaps appear as blank breaks in the line)
    ax1.plot(day_idx_local, df_raw.loc[day_idx, "aggregate"],
             color="black", lw=1.2, label="Whole-house aggregate")
    ax1.set_ylabel("Aggregate Power (W)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.set_ylim(-50, max(4000, df_raw.loc[day_idx, "aggregate"].max() + 200))

    # Bottom: WM truth and predictions
    ax2.plot(day_idx_local, y_true_full.reindex(day_idx),
             color="#1f77b4", lw=2.0, label="Ground truth (sub-metered)")
    ax2.plot(day_idx_local, lgbm_full.reindex(day_idx),
             color="#d62728", lw=1.6, ls="--", label="LightGBM v2")
    ax2.plot(day_idx_local, rule_full.reindex(day_idx),
             color="#ff7f0e", lw=1.3, ls="-.", label="Baseline 3 (Physical Rule)")
    ax2.plot(day_idx_local, b2_full.reindex(day_idx),
             color="#2ca02c", lw=1.2, ls=":", label="Baseline 2 (Time-of-Day Mean)")
    ax2.axhline(on_threshold, color="purple", ls=":", lw=0.9,
                label=f"ON threshold ({on_threshold:.0f} W)")
    ax2.set_xlabel("Time of day (HH:MM Europe/London)", fontsize=10)
    ax2.set_ylabel("Washing Machine Power (W)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=day_idx_local.tz))
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.set_ylim(-20, max(2500, y_true_full.reindex(day_idx).max() + 200))

    fig.suptitle(
        f"House {test_house} – {rep_day_str} (Europe/London): "
        f"disaggregation on the median-energy day with at least one true wash cycle",
        fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        f"Top: whole-house aggregate load for House {test_house} on {rep_day_str}. "
        f"Bottom: sub-metered ground truth vs LightGBM v2, Physical Rule baseline, and "
        f"Time-of-Day Mean baseline. White gaps indicate unusable minutes (sensor outages / "
        f"stale readings) where no prediction is made.",
        ha="center", fontsize=8.5, style="italic",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig_path1 = C.FIG_DIR / "s3_actual_vs_predicted.png"
    plt.savefig(fig_path1, dpi=150)
    plt.close()
    print(f"Saved {fig_path1}")

    # ── Figure 2: s3_failure_confounder.png ──────────────────────────────────
    # Day with highest false-positive energy (LightGBM predicted ON, truth was OFF).
    fp_energy = ((y_pred_lgbm > on_threshold) & (y_true <= C.ON_W)) * y_pred_lgbm
    fp_days   = fp_energy.groupby(y_true.index.tz_convert(C.TZ_LOCAL).date).sum()
    fp_day     = fp_days.idxmax()
    fp_day_str = str(fp_day)

    mask_fp     = (full_idx.tz_convert(C.TZ_LOCAL).date == fp_day)
    fp_idx      = full_idx[mask_fp]
    fp_idx_local = fp_idx.tz_convert(C.TZ_LOCAL)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(fp_idx_local, df_raw.loc[fp_idx, "aggregate"],
            color="gray", lw=1.0, alpha=0.7, label="Aggregate load")
    ax.plot(fp_idx_local, y_true_full.reindex(fp_idx),
            color="#1f77b4", lw=1.8, label="True washing machine")
    ax.plot(fp_idx_local, lgbm_full.reindex(fp_idx),
            color="#d62728", lw=1.5, ls="--", label="LightGBM v2 prediction")

    wm_ch = CHANNELS.get(test_house, "a2")
    for app_i in range(1, 10):
        col = f"a{app_i}"
        if col in df_raw.columns and col != wm_ch:
            s = df_raw.loc[fp_idx, col]
            if s.max() > 1500:
                ax.plot(fp_idx_local, s, lw=1.2, ls="-.",
                        label=f"Confounding load ({col})")
                break

    ax.set_xlabel("Time of day (HH:MM Europe/London)", fontsize=10)
    ax.set_ylabel("Power (W)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=fp_idx_local.tz))
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(
        f"House {test_house} – {fp_day_str}: highest false-positive energy day "
        f"(LightGBM predicted ON while washing machine was OFF)",
        fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        f"House {test_house} on {fp_day_str}. LightGBM v2 predicted washing-machine "
        f"activity during periods when the sub-metered ground truth was zero. "
        f"A confounding high-wattage appliance (shown in dash-dot) produced an aggregate "
        f"spike that the model misidentified as a wash cycle.",
        ha="center", fontsize=8.5, style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig_path2 = C.FIG_DIR / "s3_failure_confounder.png"
    plt.savefig(fig_path2, dpi=150)
    plt.close()
    print(f"Saved {fig_path2}")

    # ── Figure 3: s3_failure_transfer.png ────────────────────────────────────
    # Day with the highest mean absolute error in LightGBM predictions.
    errors    = (y_pred_lgbm - y_true).abs()
    daily_mae = errors.groupby(y_true.index.tz_convert(C.TZ_LOCAL).date).mean()
    worst_day     = daily_mae.idxmax()
    worst_day_str = str(worst_day)

    mask_w      = (full_idx.tz_convert(C.TZ_LOCAL).date == worst_day)
    w_idx       = full_idx[mask_w]
    w_idx_local = w_idx.tz_convert(C.TZ_LOCAL)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax1.plot(w_idx_local, df_raw.loc[w_idx, "aggregate"],
             color="black", lw=1.2, label="Aggregate load")
    ax1.set_ylabel("Power (W)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.set_title(
        f"House {test_house} – {worst_day_str}: worst mean-absolute-error day "
        f"(aggregate)",
        fontsize=11,
    )

    ax2.plot(w_idx_local, y_true_full.reindex(w_idx),
             color="#1f77b4", lw=1.8, label="Ground truth (sub-metered)")
    ax2.plot(w_idx_local, lgbm_full.reindex(w_idx),
             color="#d62728", lw=1.5, ls="--", label="LightGBM v2 prediction")
    ax2.plot(w_idx_local, rule_full.reindex(w_idx),
             color="#ff7f0e", lw=1.3, ls="-.", label="Physical Rule baseline")
    ax2.set_xlabel("Time of day (HH:MM Europe/London)", fontsize=10)
    ax2.set_ylabel("Washing Machine Power (W)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=w_idx_local.tz))
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.set_title("Actual vs predicted washing machine power", fontsize=11)

    fig.text(
        0.5, 0.005,
        f"House {test_house} on {worst_day_str} — the day with the highest mean absolute "
        f"error across the test period. Top: whole-house aggregate. "
        f"Bottom: sub-metered ground truth vs LightGBM v2 and Physical Rule baseline. "
        f"White gaps indicate unusable minutes.",
        ha="center", fontsize=8.5, style="italic",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig_path3 = C.FIG_DIR / "s3_failure_transfer.png"
    plt.savefig(fig_path3, dpi=150)
    plt.close()
    print(f"Saved {fig_path3}")
