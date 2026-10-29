"""Data quality masks, availability metrics, and candidate eligibility (Phase 4)."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import config as C
from . import plotting as P
from .cycles import detect_cycles, long_runs, run_lengths


def compute_house_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute quality masks for a 1-minute cleaned house DataFrame.

    Masks:
      - present: aggregate is not NaN
      - stale: runs of >= STALE_MIN consecutive minutes where all 10 channels changed < 0.01 W
      - bad_issue: issues == 1
      - usable: present & ~stale & ~bad_issue
    """
    channels = ["aggregate"] + [f"a{i}" for i in range(1, 10)]
    present = df["aggregate"].notna()

    # All channels diff < 0.01 W
    diffs = df[channels].diff().abs()
    unchanged = (diffs < 0.01).all(axis=1) & present
    stale = long_runs(unchanged, C.STALE_MIN)

    bad_issue = (df["issues"] == 1).fillna(False).to_numpy()
    usable = present.to_numpy() & ~stale & ~bad_issue

    masks_df = pd.DataFrame(
        {
            "present": present.to_numpy(),
            "stale": stale,
            "bad_issue": bad_issue,
            "usable": usable,
        },
        index=df.index,
    )
    return masks_df


def generate_stale_check_figure():
    """Generate figures/s2_stale_check.png showing run-length histograms and flat-line trace."""
    P.set_style()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Left: Histogram of all unchanged run lengths across all 20 houses (pooled)
    all_lens = []
    channels = ["aggregate"] + [f"a{i}" for i in range(1, 10)]
    for h in range(1, 22):
        if h == 14:
            continue
        df = pd.read_parquet(C.PROC_DIR / f"house_{h}_1min.parquet")
        diffs = df[channels].diff().abs()
        unchanged = (diffs < 0.01).all(axis=1) & df["aggregate"].notna()
        lens = run_lengths(unchanged)
        all_lens.extend(lens)

    all_lens = np.array(all_lens)
    clipped_lens = np.clip(all_lens, 1, 60)
    ax1.hist(clipped_lens, bins=60, range=(1, 60), color="#1f77b4", edgecolor="black", alpha=0.7)
    ax1.axvline(15, color="green", linestyle="--", lw=2, label="STALE_MIN = 15 min (Proposed)")
    ax1.axvline(30, color="red", linestyle="--", lw=2, label="STALE_MIN = 30 min (Baseline)")
    ax1.set_title("Pooled Run Lengths of Consecutive Unchanged Minutes Across All 20 Houses")
    ax1.set_xlabel("Run Length (minutes, clipped at 60 min)")
    ax1.set_ylabel("Count of Runs")
    ax1.legend(loc="upper right")

    # Right: Observed 27-minute flat run in House 1 (2014-08-01 17:34 to 18:00 UTC) with 60 min padding
    h1_df = pd.read_parquet(C.PROC_DIR / "house_1_1min.parquet")
    window = h1_df.loc["2014-08-01 16:34:00":"2014-08-01 19:00:00"]
    
    # Plot all 10 channels
    ax2.plot(window.index, window["aggregate"], label="Aggregate", color="black", lw=1.8)
    palette = plt.cm.tab10(np.linspace(0, 1, 9))
    for i in range(1, 10):
        ch = f"a{i}"
        lbl = f"a{i} (WM)" if i == 5 else f"a{i}"
        ax2.plot(window.index, window[ch], label=lbl, color=palette[i - 1], lw=1.0, alpha=0.85)

    # Highlight flat run stretch
    ax2.axvspan(
        pd.Timestamp("2014-08-01 17:34:00+00:00"),
        pd.Timestamp("2014-08-01 18:00:00+00:00"),
        color="salmon", alpha=0.25, label="27-min Flat Run"
    )
    ax2.set_title("House 1 Trace (2014-08-01): 27-min Flat Run with ±60 min Context (All 10 Channels)")
    ax2.set_ylabel("Power (W)")
    ax2.set_xlabel("Time (UTC)")
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M\n%Y-%m-%d"))
    ax2.legend(loc="upper right", ncol=2, fontsize=7.5)

    fig.text(
        0.5, 0.01,
        "Figure s2_stale_check.png: Left: Distribution of consecutive unchanged minutes across all 10 channels pooled across all 20 houses (clipped at 60 min), with thresholds at 15 min and 30 min. Right: Observed 27-minute flat run in House 1 (2014-08-01 17:34 to 18:00 UTC) across all 10 channels with 60 minutes of data either side.",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig_path = C.FIG_DIR / "s2_stale_check.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved {fig_path}")


def generate_cycle_tuning_figure(eligible_candidates: List[Tuple[int, int, str]] = None):
    """Plot stratified sample: 2 random cycles per eligible house (seeded) so every house appears."""
    P.set_style()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)

    if eligible_candidates is None:
        eligible_candidates = [
            (1, 5, "1"), (2, 2, "2"), (3, 6, "3"), (4, 5, "4b"), (5, 3, "5"),
            (6, 2, "6"), (7, 5, "7"), (8, 4, "8"), (9, 3, "9"), (10, 5, "10"),
            (11, 3, "11"), (13, 3, "13"), (15, 3, "15"), (16, 5, "16"),
            (17, 4, "17"), (18, 5, "18"), (19, 2, "19"), (20, 4, "20"), (21, 3, "21")
        ]

    sampled_plots = []
    houses_plotted = []

    for h, ch, label in eligible_candidates:
        df = pd.read_parquet(C.PROC_DIR / f"house_{h}_1min.parquet")
        present = df["aggregate"].notna()
        diffs = df[["aggregate"] + [f"a{i}" for i in range(1, 10)]].diff().abs()
        unchanged = (diffs < 0.01).all(axis=1) & present
        stale = long_runs(unchanged, C.STALE_MIN)

        wm = df[f"a{ch}"].copy()
        wm[~present | stale] = np.nan
        if h == 13:
            wm = wm.loc[:C.H13_WM_END_UTC]

        cycles = detect_cycles(wm)
        if len(cycles) >= 2:
            idx = rng.choice(len(cycles), size=2, replace=False)
            c1, c2 = cycles.iloc[idx[0]], cycles.iloc[idx[1]]
            sampled_plots.append((h, ch, label, 1, c1, wm))
            sampled_plots.append((h, ch, label, 2, c2, wm))
            houses_plotted.append(label)
        elif len(cycles) == 1:
            sampled_plots.append((h, ch, label, 1, cycles.iloc[0], wm))
            houses_plotted.append(label)

    n_plots = len(sampled_plots)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.8 * n_rows), sharey=False)
    axes = axes.flatten()

    for i, (h, ch, label, c_num, c, wm_s) in enumerate(sampled_plots):
        ax = axes[i]
        pad = pd.Timedelta(minutes=15)
        st = c["start"] - pad
        en = c["end"] + pad
        seg = wm_s.loc[st:en]

        t_mins = (seg.index - c["start"]).total_seconds() / 60.0
        ax.plot(t_mins, seg.values, color="#1f77b4", lw=1.4)
        ax.axvline(0, color="green", linestyle="--", lw=1.2, label="Start" if i == 0 else "")
        ax.axvline(c["duration_min"], color="red", linestyle="--", lw=1.2, label="End" if i == 0 else "")
        ax.axhline(C.ON_W, color="gray", linestyle=":", lw=0.8)

        date_str = c["start"].strftime("%Y-%m-%d %H:%M")
        ax.set_title(
            f"H{label} (a{ch}) #{c_num}: {date_str} ({int(c['duration_min'])}m, {c['energy_wh']:.0f}Wh)",
            fontsize=8.5
        )
        ax.set_xlabel("Time (min)", fontsize=7.5)
        ax.set_ylabel("Power (W)", fontsize=7.5)
        ax.tick_params(labelsize=7)

    # Hide extra subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Figure s2_cycle_tuning.png: Stratified Sample of Detected Washing Machine Cycles (2 Cycles per Eligible House, N={n_plots})",
        fontsize=13, y=0.995
    )
    fig.text(
        0.5, 0.005,
        "Figure s2_cycle_tuning.png: Stratified sample of washing machine cycles across all 19 eligible houses (2 seeded random cycles per house). Green dashed line: cycle start; red dashed line: cycle end; gray dotted line: ON threshold (20 W).",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout(rect=[0, 0.015, 1, 0.99])
    fig_path = C.FIG_DIR / "s2_cycle_tuning.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved {fig_path} ({n_plots} subplots across {len(houses_plotted)} houses: {houses_plotted})")


def build_availability_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute availability and eligibility tables for all WM candidates."""
    avail_rows = []
    elig_rows = []

    # Map candidate entries (handle House 4 appearing as 4a=ch4 and 4b=ch5)
    candidates_expanded = []
    for h, ch in C.WM_CANDIDATES:
        if h == 4 and ch == 4:
            candidates_expanded.append((h, ch, "4a"))
        elif h == 4 and ch == 5:
            candidates_expanded.append((h, ch, "4b"))
        else:
            candidates_expanded.append((h, ch, str(h)))

    for h, ch, label in candidates_expanded:
        p_path = C.PROC_DIR / f"house_{h}_1min.parquet"
        df = pd.read_parquet(p_path)

        # Truncate House 13 if applicable
        notes = []
        if h == 13:
            df = df.loc[:C.H13_WM_END_UTC]
            notes.append("Truncated before 2015-03-25 equipment change")
        if h == 4:
            notes.append("Dual machine household (excluded from modelling)")

        total_mins = len(df)
        present_mask = df["aggregate"].notna().to_numpy()
        pct_present = float(present_mask.mean() * 100)

        # Stale runs
        diffs = df[["aggregate"] + [f"a{i}" for i in range(1, 10)]].diff().abs()
        unchanged = (diffs < 0.01).all(axis=1) & present_mask
        stale_mask = long_runs(unchanged, C.STALE_MIN)
        pct_stale = float(stale_mask.mean() * 100)

        # Issues
        issues_mask = (df["issues"] == 1).fillna(False).to_numpy()
        pct_issues = float(issues_mask.mean() * 100)

        # Usable minutes
        usable_mask = present_mask & ~stale_mask & ~issues_mask
        usable_mins = int(usable_mask.sum())
        usable_days = round(usable_mins / 1440.0, 1)

        # First and last usable timestamps
        usable_indices = df.index[usable_mask]
        first_usable = str(usable_indices[0]) if len(usable_indices) > 0 else "None"
        last_usable = str(usable_indices[-1]) if len(usable_indices) > 0 else "None"

        # Outages on aggregate (runs of ~present_mask)
        outage_lens = run_lengths(~present_mask)
        longest_outage_min = int(outage_lens.max()) if len(outage_lens) > 0 else 0
        n_outages_gt_60min = int((outage_lens >= 60).sum())

        # Power channel statistics
        wm_series = df[f"a{ch}"].copy()
        usable_wm = wm_series[usable_mask]
        pct_zero = float((usable_wm == 0).mean() * 100) if len(usable_wm) > 0 else 100.0
        pct_off = float((usable_wm <= C.ON_W).mean() * 100) if len(usable_wm) > 0 else 100.0

        # Cycle detection: mask true outages and stale runs
        wm_for_cycles = wm_series.copy()
        wm_for_cycles[~present_mask | stale_mask] = np.nan
        cycles = detect_cycles(wm_for_cycles)

        n_cycles = len(cycles)
        usable_weeks = usable_days / 7.0
        cycles_per_week = round(n_cycles / usable_weeks, 2) if usable_weeks > 0 else 0.0

        if n_cycles > 0:
            med_dur = round(float(cycles["duration_min"].median()), 1)
            med_wh = round(float(cycles["energy_wh"].median()), 1)
            med_peak = round(float(cycles["peak_w"].median()), 1)
        else:
            med_dur, med_wh, med_peak = 0.0, 0.0, 0.0

        # Eligibility evaluation
        pass_days = usable_days >= C.MIN_USABLE_DAYS
        pass_cycles = n_cycles >= C.MIN_CYCLES
        pass_dur = C.SIG_DUR_RANGE_MIN[0] <= med_dur <= C.SIG_DUR_RANGE_MIN[1]
        pass_peak = med_peak >= C.SIG_MIN_PEAK_W
        pass_signature = pass_dur and pass_peak
        eligible = pass_days and pass_cycles and pass_signature

        fail_reasons = []
        if not pass_days:
            fail_reasons.append(f"Usable days ({usable_days}) < {C.MIN_USABLE_DAYS}")
        if not pass_cycles:
            fail_reasons.append(f"Cycle count ({n_cycles}) < {C.MIN_CYCLES}")
        if not pass_dur:
            fail_reasons.append(f"Median duration ({med_dur}m) outside {C.SIG_DUR_RANGE_MIN}")
        if not pass_peak:
            fail_reasons.append(f"Median peak ({med_peak}W) < {C.SIG_MIN_PEAK_W}W")

        status_str = "Eligible" if eligible else "Excluded: " + "; ".join(fail_reasons)

        avail_rows.append({
            "house": label,
            "channel": f"a{ch}",
            "first_usable_utc": first_usable,
            "last_usable_utc": last_usable,
            "usable_days": usable_days,
            "pct_present": round(pct_present, 2),
            "pct_stale": round(pct_stale, 2),
            "pct_issues": round(pct_issues, 2),
            "longest_outage_min": longest_outage_min,
            "n_outages_gt_60min": n_outages_gt_60min,
            "pct_zero": round(pct_zero, 2),
            "pct_off": round(pct_off, 2),
            "n_cycles": n_cycles,
            "median_duration_min": med_dur,
            "median_energy_wh": med_wh,
            "median_peak_w": med_peak,
            "cycles_per_week": cycles_per_week,
            "occupancy": C.OCCUPANCY.get(h, "Unknown"),
            "appliances_owned": C.APPLIANCES_OWNED.get(h, "Unknown"),
            "has_tumble_dryer": h in C.TUMBLE_DRYER,
            "has_washer_dryer": h in C.WASHER_DRYER,
            "notes": ", ".join(notes) if notes else "",
        })

        elig_rows.append({
            "house": label,
            "channel": f"a{ch}",
            "usable_days": usable_days,
            "pass_days": pass_days,
            "n_cycles": n_cycles,
            "pass_cycles": pass_cycles,
            "median_duration_min": med_dur,
            "pass_duration": pass_dur,
            "median_peak_w": med_peak,
            "pass_peak": pass_peak,
            "eligible": eligible,
            "status": status_str,
        })

    avail_df = pd.DataFrame(avail_rows)
    elig_df = pd.DataFrame(elig_rows)

    # Save to outputs/
    avail_csv_path = C.OUT_DIR / "availability_clean.csv"
    elig_csv_path = C.OUT_DIR / "eligibility.csv"

    avail_df.to_csv(avail_csv_path, index=False)
    elig_df.to_csv(elig_csv_path, index=False)
    print(f"Saved availability table to {avail_csv_path}")
    print(f"Saved eligibility table to {elig_csv_path}")

    return avail_df, elig_df


def stage_quality():
    """Execute complete Phase 4 pipeline."""
    print("=" * 60)
    print("STAGE: Quality Masks, Cycle Tuning, Availability & Eligibility")
    print("=" * 60)
    t0 = time.time()

    # 1. Stale check figure
    print("Generating stale check figure...")
    generate_stale_check_figure()

    # 2. Availability and Eligibility tables
    print("Building availability and eligibility tables...")
    avail_df, elig_df = build_availability_table()

    # 3. Cycle tuning figure (stratified: 2 cycles per eligible house)
    print("Generating cycle tuning figure...")
    generate_cycle_tuning_figure()

    print(f"\nStage Quality completed in {time.time() - t0:.1f}s")
    print("=" * 60)
    return avail_df, elig_df
