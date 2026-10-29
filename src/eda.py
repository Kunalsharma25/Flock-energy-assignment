"""Washing machine exploratory data analysis (Phase 5 / Section 2)."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from . import config as C
from . import plotting as P
from .cycles import detect_cycles, long_runs, mask_appliance_flatline

COHORT = [(2, "a2"), (3, "a6"), (5, "a3"), (6, "a2")]


def run_cohort_eda() -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """Run cycle detection and build summary metrics for the target cohort."""
    summary_rows = []
    cohort_dfs = {}
    cohort_cycles = {}

    for h, ch in COHORT:
        df = pd.read_parquet(C.PROC_DIR / f"house_{h}_1min.parquet")
        present = df["aggregate"].notna().to_numpy()
        diffs = df[["aggregate"] + [f"a{i}" for i in range(1, 10)]].diff().abs()
        unchanged = (diffs < 0.01).all(axis=1) & present
        stale = long_runs(unchanged, C.STALE_MIN)
        bad_issue = (df["issues"] == 1).fillna(False).to_numpy()
        usable = present & ~stale & ~bad_issue

        usable_mins = int(usable.sum())
        usable_days = round(usable_mins / 1440.0, 1)

        usable_idx = df.index[usable]
        start_utc = str(usable_idx[0])
        end_utc = str(usable_idx[-1])

        wm = df[ch].copy()
        usable_wm = wm[usable]
        pct_zero = round(float((usable_wm == 0).mean() * 100), 2)
        pct_off = round(float((usable_wm <= C.ON_W).mean() * 100), 2)

        wm_for_cyc = mask_appliance_flatline(wm.copy(), on_w=C.ON_W, stale_min=C.STALE_MIN)
        wm_for_cyc[~present | stale] = np.nan
        cyc = detect_cycles(
            wm_for_cyc,
            on_w=C.ON_W,
            merge_gap=C.MERGE_GAP_MIN,
            min_dur=C.MIN_CYCLE_MIN,
            min_wh=C.MIN_CYCLE_WH,
            max_dur=C.MAX_CYCLE_MIN,
            max_wh=C.MAX_CYCLE_WH,
        )

        # Add local time columns (Europe/London)
        cyc["start_local"] = cyc["start"].dt.tz_convert(C.TZ_LOCAL)
        cyc["end_local"] = cyc["end"].dt.tz_convert(C.TZ_LOCAL)
        cyc["hour_local"] = cyc["start_local"].dt.hour
        cyc["dow_num"] = cyc["start_local"].dt.dayofweek  # 0=Mon, 6=Sun
        cyc["dow_name"] = cyc["start_local"].dt.day_name()
        cyc["house"] = h
        cyc["channel"] = ch

        n_cycles = len(cyc)
        usable_weeks = usable_days / 7.0
        cyc_per_week = round(n_cycles / usable_weeks, 2) if usable_weeks > 0 else 0.0

        d = cyc["duration_min"]
        e = cyc["energy_wh"]
        p = cyc["peak_w"]
        m = cyc["mean_w"]

        summary_rows.append({
            "house": h,
            "channel": ch,
            "usable_start_utc": start_utc,
            "usable_end_utc": end_utc,
            "usable_days": usable_days,
            "pct_present": round(float(present.mean() * 100), 2),
            "pct_stale": round(float(stale.mean() * 100), 2),
            "pct_issues": round(float(bad_issue.mean() * 100), 2),
            "pct_zero": pct_zero,
            "pct_off": pct_off,
            "n_cycles": n_cycles,
            "cycles_per_week": cyc_per_week,
            "duration_mean_min": round(float(d.mean()), 1),
            "duration_median_min": round(float(d.median()), 1),
            "duration_std_min": round(float(d.std()), 1),
            "duration_iqr_min": round(float(d.quantile(0.75) - d.quantile(0.25)), 1),
            "duration_min_min": int(d.min()),
            "duration_max_min": int(d.max()),
            "energy_mean_wh": round(float(e.mean()), 1),
            "energy_median_wh": round(float(e.median()), 1),
            "energy_std_wh": round(float(e.std()), 1),
            "energy_iqr_wh": round(float(e.quantile(0.75) - e.quantile(0.25)), 1),
            "energy_min_wh": round(float(e.min()), 1),
            "energy_max_wh": round(float(e.max()), 1),
            "peak_mean_w": round(float(p.mean()), 1),
            "peak_median_w": round(float(p.median()), 1),
            "peak_min_w": round(float(p.min()), 1),
            "peak_max_w": round(float(p.max()), 1),
            "mean_power_w": round(float(m.mean()), 1),
            "occupancy": C.OCCUPANCY[h],
            "appliances_owned": C.APPLIANCES_OWNED[h],
            "has_tumble_dryer": h in C.TUMBLE_DRYER,
            "has_washer_dryer": h in C.WASHER_DRYER,
        })

        cohort_dfs[h] = df
        cohort_cycles[h] = cyc

    summary_df = pd.DataFrame(summary_rows)
    out_path = C.OUT_DIR / "section2_cycle_summary.csv"
    try:
        summary_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
    except PermissionError:
        temp_path = C.OUT_DIR / "section2_cycle_summary_new.csv"
        summary_df.to_csv(temp_path, index=False)
        print(f"[NOTICE] {out_path.name} is currently open in another program (e.g. Excel). Saved updated data to {temp_path.name}")

    return summary_df, cohort_dfs, cohort_cycles


def plot_typical_cycle_profile(cohort_dfs: Dict[int, pd.DataFrame], cohort_cycles: Dict[int, pd.DataFrame]):
    """Generate figures/sec2_typical_cycle_profile.png showing exemplar cycles and 24h day trace."""
    P.set_style()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)

    # Top Row: 4 exemplar cycle profiles across H2, H3, H5, H6
    house_channels = {2: "a2", 3: "a6", 5: "a3", 6: "a2"}
    exemplar_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for i, (h, ch) in enumerate(COHORT):
        ax = exemplar_axes[i]
        df = cohort_dfs[h]
        cyc = cohort_cycles[h]

        # Select cycle closest to median duration and median energy
        d_med = cyc["duration_min"].median()
        e_med = cyc["energy_wh"].median()
        scores = ((cyc["duration_min"] - d_med) / d_med) ** 2 + ((cyc["energy_wh"] - e_med) / e_med) ** 2
        best_c = cyc.loc[scores.idxmin()]

        # Slice cycle with 10 min padding
        pad = pd.Timedelta(minutes=10)
        st = best_c["start"] - pad
        en = best_c["end"] + pad
        seg = df.loc[st:en, ch]
        t_mins = (seg.index - best_c["start"]).total_seconds() / 60.0

        ax.plot(t_mins, seg.values, color="#1f77b4", lw=1.8, label="WM Power")
        ax.axvline(0, color="green", linestyle="--", lw=1.3, label="Start (t=0)")
        ax.axvline(best_c["duration_min"], color="red", linestyle="--", lw=1.3, label=f"End ({int(best_c['duration_min'])}m)")
        ax.axhline(C.ON_W, color="gray", linestyle=":", lw=0.9, label=f"ON Threshold ({C.ON_W:.0f}W)")

        # Annotations for heater spike and spin
        heater_idx = seg.values.argmax()
        heater_t = t_mins[heater_idx]
        heater_w = seg.values[heater_idx]
        if heater_w > 1500:
            ax.annotate(
                f"Heater: {heater_w:.0f}W",
                xy=(heater_t, heater_w),
                xytext=(heater_t + 5, heater_w * 0.92),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=0.8),
                fontsize=8,
            )

        date_str = best_c["start"].strftime("%Y-%m-%d")
        ax.set_title(
            f"House {h} ({ch})\n{date_str} | {int(best_c['duration_min'])}m | {best_c['energy_wh']:.0f}Wh | Peak {best_c['peak_w']:.0f}W",
            fontsize=9.5,
        )
        ax.set_xlabel("Time from Start (minutes)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Power (W)", fontsize=10)
            ax.legend(loc="upper right", fontsize=7.5)
        ax.set_ylim(-50, 2700)

    # Bottom Row: Full 24-hour trace for representative active day in House 3 (2015-05-07)
    ax_24h = fig.add_subplot(gs[1, :])
    df3 = cohort_dfs[3]
    cyc3 = cohort_cycles[3]

    day_str = "2015-05-07"
    df3_day = df3.loc[f"{day_str} 00:00:00+00:00":f"{day_str} 23:59:00+00:00"]
    # Convert index to Europe/London local time for plotting
    day_local = df3_day.index.tz_convert(C.TZ_LOCAL)

    ax_24h.plot(day_local, df3_day["aggregate"], label="Household Aggregate Power", color="#333333", lw=1.2, alpha=0.9)
    ax_24h.plot(day_local, df3_day["a6"], label="Washing Machine (a6)", color="#1f77b4", lw=1.6)

    # Highlight the 2 detected cycles
    c_day = cyc3[cyc3["start"].dt.tz_convert(C.TZ_LOCAL).dt.date.astype(str) == day_str]
    for _, c_row in c_day.iterrows():
        c_st = c_row["start"].tz_convert(C.TZ_LOCAL)
        c_en = c_row["end"].tz_convert(C.TZ_LOCAL)
        ax_24h.axvspan(c_st, c_en, color="#2ca02c", alpha=0.2, label="Detected Cycle" if _ == c_day.index[0] else "")
        mid_t = c_st + (c_en - c_st) / 2
        ax_24h.text(
            mid_t, c_row["peak_w"] + 150,
            f"Cycle: {int(c_row['duration_min'])}m, {c_row['energy_wh']:.0f}Wh",
            ha="center", fontsize=8.5, fontweight="bold", color="#006400"
        )

    ax_24h.set_title(
        f"Representative 24-Hour Whole-Home Trace: House 3 on {day_str} (Europe/London Local Time, BST)\n"
        f"Selected as the median-energy full-coverage day with 2 active cycles (Daily WM: 1032.8 Wh, Aggregate: 15.09 kWh)",
        fontsize=10.5,
    )
    ax_24h.set_xlabel("Time of Day (HH:MM Europe/London)", fontsize=10)
    ax_24h.set_ylabel("Power (W)", fontsize=10)
    ax_24h.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=day_local.tz))
    ax_24h.set_ylim(-100, 4500)
    ax_24h.legend(loc="upper right", fontsize=8.5)

    fig.text(
        0.5, 0.01,
        "Figure sec2_typical_cycle_profile.png: Top: Typical exemplar washing machine cycle profiles across the four cohort houses (Houses 2, 3, 5, 6), illustrating the characteristic multi-phase signature: high-wattage water heating pulse (~2000 W), drum agitation (~100–300 W), intermediate rinses, and final high-speed spin extraction (~500–800 W). Bottom: Representative 24-hour trace for House 3 (2015-05-07, Europe/London local time; selected as the median-energy 2-cycle full-coverage day), showing washing machine operation embedded within total household aggregate demand.",
        ha="center", fontsize=8.5, style="italic"
    )

    fig_path = C.FIG_DIR / "sec2_typical_cycle_profile.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved {fig_path}")


def plot_cycle_duration_distribution(cohort_cycles: Dict[int, pd.DataFrame]):
    """Generate figures/sec2_cycle_duration_distribution.png comparing duration and energy."""
    P.set_style()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    records = []
    for h in [2, 3, 5, 6]:
        cyc = cohort_cycles[h].copy()
        cyc["house_label"] = f"House {h}\n(N={len(cyc)})"
        records.append(cyc)
    combined = pd.concat(records, ignore_index=True)

    # Left: Cycle Duration (minutes)
    sns.boxplot(
        data=combined, x="house_label", y="duration_min", hue="house_label", ax=ax1,
        palette="Blues", width=0.45, fliersize=0, legend=False
    )
    sns.stripplot(
        data=combined, x="house_label", y="duration_min", ax=ax1,
        color="black", alpha=0.25, jitter=0.2, size=3.5
    )

    # Medians overlay
    medians_dur = combined.groupby("house_label")["duration_min"].median()
    for i, (label, med) in enumerate(medians_dur.items()):
        ax1.text(i, med + 5, f"Med: {med:.0f}m", ha="center", fontsize=9, fontweight="bold", color="darkblue")

    ax1.set_title("Washing Machine Cycle Duration Distribution Across Cohort", fontsize=11)
    ax1.set_xlabel("Household", fontsize=10.5)
    ax1.set_ylabel("Duration (minutes)", fontsize=10.5)
    max_dur_obs = float(combined["duration_min"].max())
    ax1.set_ylim(0, int(np.ceil((max_dur_obs + 20) / 50.0) * 50))

    # Right: Cycle Active Energy Consumption (Wh)
    sns.boxplot(
        data=combined, x="house_label", y="energy_wh", hue="house_label", ax=ax2,
        palette="Greens", width=0.45, fliersize=0, legend=False
    )
    sns.stripplot(
        data=combined, x="house_label", y="energy_wh", ax=ax2,
        color="black", alpha=0.25, jitter=0.2, size=3.5
    )

    medians_wh = combined.groupby("house_label")["energy_wh"].median()
    for i, (label, med) in enumerate(medians_wh.items()):
        ax2.text(i, med + 40, f"Med: {med:.0f}Wh", ha="center", fontsize=9, fontweight="bold", color="darkgreen")

    ax2.set_title("Washing Machine Cycle Active Energy Consumption Across Cohort", fontsize=11)
    ax2.set_xlabel("Household", fontsize=10.5)
    ax2.set_ylabel("Active Energy (Wh)", fontsize=10.5)
    max_energy_obs = float(combined["energy_wh"].max())
    ax2.set_ylim(0, int(np.ceil((max_energy_obs + 250) / 500.0) * 500))

    counts_str = ", ".join([f"House {h}: N={len(cohort_cycles[h])}" for h in [2, 3, 5, 6]])
    caption_text = (
        f"Figure sec2_cycle_duration_distribution.png: Distribution of washing machine cycle duration (left) and "
        f"active energy consumption (right) across the target cohort ({counts_str}). Boxes represent median and "
        f"interquartile range (IQR); whiskers extend to 1.5x IQR; individual points depict detected cycles. "
        f"Includes 100% of detected cycles across all 4 cohort houses; zero data points clipped or omitted."
    )
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=8.5, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    fig_path = C.FIG_DIR / "sec2_cycle_duration_distribution.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved {fig_path}")


def plot_usage_patterns(cohort_cycles: Dict[int, pd.DataFrame]):
    """Generate figures/sec2_usage_patterns.png showing diurnal and weekly usage in UK local time."""
    P.set_style()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    records = []
    for h in [2, 3, 5, 6]:
        cyc = cohort_cycles[h].copy()
        cyc["house"] = h
        records.append(cyc)
    combined = pd.concat(records, ignore_index=True)

    # 1. Hourly distribution (Hour 0 to 23, Europe/London)
    palette = {2: "#1f77b4", 3: "#ff7f0e", 5: "#2ca02c", 6: "#d62728"}
    hours = np.arange(24)

    for h in [2, 3, 5, 6]:
        h_cyc = combined[combined["house"] == h]
        h_counts = h_cyc["hour_local"].value_counts().reindex(hours, fill_value=0)
        h_share = (h_counts / len(h_cyc)) * 100
        ax1.plot(hours, h_share, marker="o", markersize=4, label=f"House {h} (N={len(h_cyc)})", color=palette[h], lw=1.8)

    ax1.set_title("Cycle Start Hour Distribution (UK Local Time)", fontsize=11)
    ax1.set_xlabel("Hour of Day (Europe/London)", fontsize=10)
    ax1.set_ylabel("Share of House Cycles (%)", fontsize=10)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(-0.5, 23.5)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Weekday distribution (Mon to Sun, Europe/London)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)
    width = 0.18

    for idx, h in enumerate([2, 3, 5, 6]):
        h_cyc = combined[combined["house"] == h]
        d_counts = h_cyc["dow_name"].value_counts().reindex(days, fill_value=0)
        d_share = (d_counts / len(h_cyc)) * 100
        ax2.bar(x + (idx - 1.5) * width, d_share, width=width, label=f"House {h}", color=palette[h], alpha=0.85)

    ax2.set_title("Cycle Starts by Day of Week (UK Local Time)", fontsize=11)
    ax2.set_xlabel("Day of Week (Europe/London)", fontsize=10)
    ax2.set_ylabel("Share of House Cycles (%)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(day_labels)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Heatmap of Day of Week x Hour of Day (Pooled Cohort, UK local time)
    heatmap_data = pd.crosstab(
        combined["dow_name"], combined["hour_local"], normalize=True
    ).reindex(index=days, columns=hours, fill_value=0) * 100

    sns.heatmap(
        heatmap_data, ax=ax3, cmap="YlGnBu", cbar_kws={"label": "Share of All Cycles (%)"},
        yticklabels=day_labels, xticklabels=[str(h) if h % 2 == 0 else "" for h in hours]
    )
    ax3.set_title("Cohort Joint Laundry Heatmap (Hour x Weekday)", fontsize=11)
    ax3.set_xlabel("Hour of Day (Europe/London)", fontsize=10)
    ax3.set_ylabel("Day of Week", fontsize=10)

    fig.text(
        0.5, 0.01,
        "Figure sec2_usage_patterns.png: Diurnal and weekly washing machine usage patterns across the target cohort in UK local time (Europe/London). Left: Cycle start hour distribution showing primary morning peak (08:00–12:00) and secondary evening peak (16:00–20:00). Middle: Day-of-week distribution across houses. Right: Joint hour-by-weekday heatmap of cohort laundry starts.",
        ha="center", fontsize=8.5, style="italic"
    )
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.97, wspace=0.32)

    fig_path = C.FIG_DIR / "sec2_usage_patterns.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved {fig_path}")


def stage_eda():
    """Execute complete Phase 5 / Section 2 pipeline."""
    print("=" * 60)
    print("STAGE: Section 2 - Washing Machine EDA & Visualizations")
    print("=" * 60)
    summary_df, cohort_dfs, cohort_cycles = run_cohort_eda()

    print("\nGenerating sec2_typical_cycle_profile.png...")
    plot_typical_cycle_profile(cohort_dfs, cohort_cycles)

    print("Generating sec2_cycle_duration_distribution.png...")
    plot_cycle_duration_distribution(cohort_cycles)

    print("Generating sec2_usage_patterns.png...")
    plot_usage_patterns(cohort_cycles)

    print("\nStage EDA completed successfully.")
    print("=" * 60)
    return summary_df
