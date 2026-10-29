"""Audit and verify Phase 4 calculations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import py7zr

from src import config as C
from src.cycles import detect_cycles, long_runs, run_lengths

def main():
    print("=" * 70)
    print("PHASE 4 COMPREHENSIVE AUDIT")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Flat runs across all 20 houses (pooled)
    # -------------------------------------------------------------
    print("\n--- 1. POOLED UNCHANGED RUN LENGTHS ---")
    channels = ["aggregate"] + [f"a{i}" for i in range(1, 10)]
    all_runs = []
    
    total_minutes_all = 0
    total_stale_15 = 0
    total_stale_30 = 0

    house_flat_runs = {}

    for h in range(1, 22):
        if h == 14:
            continue
        p = C.PROC_DIR / f"house_{h}_1min.parquet"
        df = pd.read_parquet(p)
        total_minutes_all += len(df)
        
        diffs = df[channels].diff().abs()
        unchanged = (diffs < 0.01).all(axis=1) & df["aggregate"].notna()
        
        # Stale masks at 15 and 30
        stale15 = long_runs(unchanged, 15)
        stale30 = long_runs(unchanged, 30)
        total_stale_15 += stale15.sum()
        total_stale_30 += stale30.sum()

        # Run lengths
        d = np.diff(np.concatenate([[0], unchanged.astype(np.int8), [0]]))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        lens = ends - starts
        all_runs.extend(lens)

        # Store runs with metadata for inspection
        runs_info = []
        for s, e, l in zip(starts, ends, lens):
            runs_info.append((df.index[s], df.index[e - 1], l, s, e))
        house_flat_runs[h] = runs_info

    all_runs = np.array(all_runs)

    r_1_2 = int(((all_runs >= 1) & (all_runs <= 2)).sum())
    r_3_10 = int(((all_runs >= 3) & (all_runs <= 10)).sum())
    r_11_29 = int(((all_runs >= 11) & (all_runs <= 29)).sum())
    r_30_59 = int(((all_runs >= 30) & (all_runs <= 59)).sum())
    r_60_plus = int((all_runs >= 60).sum())
    r_total = len(all_runs)

    print(f"Total flat runs detected: {r_total}")
    print(f"  1-2 min:   {r_1_2:6d} ({r_1_2 / r_total * 100:.2f}%)")
    print(f"  3-10 min:  {r_3_10:6d} ({r_3_10 / r_total * 100:.2f}%)")
    print(f"  11-29 min: {r_11_29:6d} ({r_11_29 / r_total * 100:.2f}%)")
    print(f"  30-59 min: {r_30_59:6d} ({r_30_59 / r_total * 100:.2f}%)")
    print(f"  60+ min:   {r_60_plus:6d} ({r_60_plus / r_total * 100:.2f}%)")
    print(f"Total flagged minutes at STALE_MIN=15: {total_stale_15:,} ({total_stale_15 / 1440:.1f} days, {total_stale_15 / total_minutes_all * 100:.2f}%)")
    print(f"Total flagged minutes at STALE_MIN=30: {total_stale_30:,} ({total_stale_30 / 1440:.1f} days, {total_stale_30 / total_minutes_all * 100:.2f}%)")
    print(f"Difference (flagged at 15 but not 30): {total_stale_15 - total_stale_30:,} minutes ({(total_stale_15 - total_stale_30) / 1440:.1f} days)")

    # -------------------------------------------------------------
    # Inspection of 20-29 minute runs
    # -------------------------------------------------------------
    print("\n--- INSPECTION OF 20-29 MINUTE RUNS ---")
    runs_20_29 = []
    for h, r_list in house_flat_runs.items():
        for st, en, l, s, e in r_list:
            if 20 <= l <= 29:
                runs_20_29.append((h, st, en, l))
    print(f"Number of 20-29 min runs across all houses: {len(runs_20_29)}")
    for item in runs_20_29[:10]:
        print(f"  House {item[0]}: {item[1]} to {item[2]} (duration {item[3]} min)")

    # Inspect House 1 27-min run
    df_h1 = pd.read_parquet(C.PROC_DIR / "house_1_1min.parquet")
    h1_slice = df_h1.loc["2014-08-01 17:34:00":"2014-08-01 18:00:00"]
    print(f"\nHouse 1 run 2014-08-01 17:34:00 to 18:00:00 (length = {len(h1_slice)} min):")
    print("Channel value std deviations across this 27-min stretch:")
    for col in channels:
        print(f"  {col}: mean={h1_slice[col].mean():.2f} W, std={h1_slice[col].std():.6f}")

    # -------------------------------------------------------------
    # 2. Eligibility & Availability comparison: STALE_MIN=15 vs 30
    # -------------------------------------------------------------
    print("\n--- 2. ELIGIBILITY: STALE_MIN=15 vs 30 ---")
    candidates_expanded = []
    for h, ch in C.WM_CANDIDATES:
        if h == 4 and ch == 4:
            candidates_expanded.append((h, ch, "4a"))
        elif h == 4 and ch == 5:
            candidates_expanded.append((h, ch, "4b"))
        else:
            candidates_expanded.append((h, ch, str(h)))

    for stale_thresh in [30, 15]:
        print(f"\nEvaluating with STALE_MIN = {stale_thresh}:")
        elig_list = []
        for h, ch, label in candidates_expanded:
            df = pd.read_parquet(C.PROC_DIR / f"house_{h}_1min.parquet")
            if h == 13:
                df = df.loc[:C.H13_WM_END_UTC]
            present = df["aggregate"].notna().to_numpy()
            diffs = df[channels].diff().abs()
            unchanged = (diffs < 0.01).all(axis=1) & present
            stale = long_runs(unchanged, stale_thresh)
            issues = (df["issues"] == 1).fillna(False).to_numpy()
            usable = present & ~stale & ~issues
            usable_days = usable.sum() / 1440.0

            wm = df[f"a{ch}"].copy()
            wm[~present | stale] = np.nan
            cycles = detect_cycles(wm)
            n_cycles = len(cycles)
            med_dur = float(cycles["duration_min"].median()) if n_cycles > 0 else 0.0
            med_peak = float(cycles["peak_w"].median()) if n_cycles > 0 else 0.0

            pass_days = usable_days >= C.MIN_USABLE_DAYS
            pass_cycles = n_cycles >= C.MIN_CYCLES
            pass_dur = C.SIG_DUR_RANGE_MIN[0] <= med_dur <= C.SIG_DUR_RANGE_MIN[1]
            pass_peak = med_peak >= C.SIG_MIN_PEAK_W
            eligible = pass_days and pass_cycles and pass_dur and pass_peak

            elig_list.append((label, ch, usable_days, n_cycles, med_dur, med_peak, eligible))

        elig_df_temp = pd.DataFrame(elig_list, columns=["house", "ch", "usable_days", "n_cycles", "med_dur", "med_peak", "eligible"])
        print(f"Eligible count at STALE_MIN={stale_thresh}: {elig_df_temp['eligible'].sum()} / {len(elig_df_temp)}")
        print("Eligible houses:", elig_df_temp[elig_df_temp["eligible"]]["house"].tolist())

    # -------------------------------------------------------------
    # 4. Cycle Discard Analysis & Issues==1 on ON vs OFF minutes
    # -------------------------------------------------------------
    print("\n--- 4. CYCLE DISCARD ANALYSIS & ISSUES==1 IN ON VS OFF ---")
    stale_thresh = 15
    discard_stats = []
    issues_on_off = []

    for h, ch, label in candidates_expanded:
        df = pd.read_parquet(C.PROC_DIR / f"house_{h}_1min.parquet")
        if h == 13:
            df = df.loc[:C.H13_WM_END_UTC]
        
        present = df["aggregate"].notna()
        diffs = df[channels].diff().abs()
        unchanged = (diffs < 0.01).all(axis=1) & present
        stale = long_runs(unchanged, stale_thresh)
        bad_issue = (df["issues"] == 1).fillna(False)
        usable = present & ~stale & ~bad_issue

        wm = df[f"a{ch}"].copy()

        # ON vs OFF issues share on present data
        valid_data = present & ~stale
        wm_valid = wm[valid_data]
        issues_valid = bad_issue[valid_data]

        on_mask = (wm_valid > C.ON_W)
        off_mask = (wm_valid <= C.ON_W)

        n_on = on_mask.sum()
        n_off = off_mask.sum()
        n_on_issues = (on_mask & issues_valid).sum()
        n_off_issues = (off_mask & issues_valid).sum()

        pct_issues_on = (n_on_issues / n_on * 100) if n_on > 0 else 0.0
        pct_issues_off = (n_off_issues / n_off * 100) if n_off > 0 else 0.0

        issues_on_off.append({
            "house": label,
            "ch": f"a{ch}",
            "n_on_min": int(n_on),
            "n_on_issues": int(n_on_issues),
            "pct_issues_on": round(pct_issues_on, 2),
            "n_off_min": int(n_off),
            "n_off_issues": int(n_off_issues),
            "pct_issues_off": round(pct_issues_off, 2),
            "ratio_on_to_off": round(pct_issues_on / pct_issues_off, 2) if pct_issues_off > 0 else 0.0
        })

        # Cycle discard analysis
        # First, detect candidate raw ON runs (using only wm > ON_W and MERGE_GAP_MIN)
        # without any NaN masking, so we find ALL raw candidate cycles
        wm_raw = wm.copy()
        on_raw = (wm_raw > C.ON_W).fillna(False).to_numpy()
        d = np.diff(np.concatenate([[0], on_raw.astype(np.int8), [0]]))
        runs = [[s, e] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
        merged = []
        for s, e in runs:
            if merged and (s - merged[-1][1]) <= C.MERGE_GAP_MIN:
                merged[-1][1] = e
            else:
                merged.append([s, e])

        v = wm_raw.fillna(0).to_numpy()
        nan_pres_stale = (~present | stale).to_numpy()
        nan_issues = bad_issue.to_numpy()
        nan_unusable = (~usable).to_numpy()

        n_candidate = 0
        n_discard_outage_stale = 0
        n_discard_issues = 0
        n_discard_any_unusable = 0

        for s, e in merged:
            dur = e - s
            seg = v[s:e]
            wh = float(seg.sum() / 60.0)
            if dur >= C.MIN_CYCLE_MIN and wh >= C.MIN_CYCLE_WH:
                n_candidate += 1
                
                # Check touches
                touch_outage_stale = (nan_pres_stale[max(s-1, 0)] or nan_pres_stale[min(e, len(nan_pres_stale)-1)] or nan_pres_stale[s:e].any())
                touch_issues = (nan_issues[max(s-1, 0)] or nan_issues[min(e, len(nan_issues)-1)] or nan_issues[s:e].any())
                touch_unusable = (nan_unusable[max(s-1, 0)] or nan_unusable[min(e, len(nan_unusable)-1)] or nan_unusable[s:e].any())

                if touch_outage_stale:
                    n_discard_outage_stale += 1
                if touch_issues:
                    n_discard_issues += 1
                if touch_unusable:
                    n_discard_any_unusable += 1

        pct_discard_outage_stale = (n_discard_outage_stale / n_candidate * 100) if n_candidate > 0 else 0.0
        pct_discard_issues = (n_discard_issues / n_candidate * 100) if n_candidate > 0 else 0.0
        pct_discard_unusable = (n_discard_any_unusable / n_candidate * 100) if n_candidate > 0 else 0.0

        discard_stats.append({
            "house": label,
            "ch": f"a{ch}",
            "candidate_cycles": n_candidate,
            "discard_outage_stale": n_discard_outage_stale,
            "pct_discard_outage_stale": round(pct_discard_outage_stale, 2),
            "discard_issues": n_discard_issues,
            "pct_discard_issues": round(pct_discard_issues, 2),
            "discard_unusable": n_discard_any_unusable,
            "pct_discard_unusable": round(pct_discard_unusable, 2),
            "retained_clean_cycles": n_candidate - n_discard_outage_stale
        })

    discard_df = pd.DataFrame(discard_stats)
    issues_on_off_df = pd.DataFrame(issues_on_off)

    print("\n--- Cycle Discard Summary ---")
    print(discard_df.to_string(index=False))

    print("\n--- Issues==1 in ON vs OFF Summary ---")
    print(issues_on_off_df.to_string(index=False))

    # -------------------------------------------------------------
    # 5. Core Audit Checks
    # -------------------------------------------------------------
    print("\n--- 5. CORE AUDIT CHECKS ---")
    
    # 5.1 tz_check.txt
    print("\n(1) outputs/tz_check.txt contents:")
    tz_path = C.OUT_DIR / "tz_check.txt"
    if tz_path.exists():
        print(tz_path.read_text().strip())
    else:
        print("tz_check.txt not found!")

    # 5.2 Header lines
    print("\n(2) Header line of RAW House 1 (Part 1):")
    raw_p1 = C.RAW_DIR / "RAW_House1_Part1.csv"
    with open(raw_p1, "r") as f:
        print(f.readline().strip())

    print("\nHeader line of CLEAN House 1:")
    with py7zr.SevenZipFile(C.ARCHIVE_DIR / C.ARCHIVES["clean"]["file"], mode="r") as z:
        # Extract first line of House 1
        z.extract(targets=["CLEAN_House1.csv"], path=C.TMP_DIR)
        with open(C.TMP_DIR / "CLEAN_House1.csv", "r") as f:
            print(f.readline().strip())
        (C.TMP_DIR / "CLEAN_House1.csv").unlink()

    # 5.3 Issues column dtype
    p_h1 = C.PROC_DIR / "house_1_1min.parquet"
    df_h1 = pd.read_parquet(p_h1)
    print(f"\n(3) 'issues' column dtype in house_1_1min.parquet: {df_h1['issues'].dtype}")

    # 5.4 Archive solid flag
    with py7zr.SevenZipFile(C.ARCHIVE_DIR / C.ARCHIVES["clean"]["file"], mode="r") as z:
        print(f"\n(4) Cleaned archive solid flag: {z.archiveinfo().solid}")

    # 5.5 Seasonal evening-peak timezone check on data/processed/house_*_1min.parquet
    print("\n(5) Seasonal evening-peak timezone check across all 20 houses:")
    summer_winter_profiles = []
    for h in range(1, 22):
        if h == 14:
            continue
        p = C.PROC_DIR / f"house_{h}_1min.parquet"
        df = pd.read_parquet(p, columns=["aggregate"])
        
        # Winter months: Dec, Jan, Feb (GMT / UTC+0)
        # Summer months: Jun, Jul, Aug (BST / UTC+1)
        winter = df[df.index.month.isin([12, 1, 2])]["aggregate"]
        summer = df[df.index.month.isin([6, 7, 8])]["aggregate"]
        
        # UTC hourly means
        w_utc = winter.groupby(winter.index.hour).mean()
        s_utc = summer.groupby(summer.index.hour).mean()

        # Local time hourly means (Europe/London)
        w_loc = winter.groupby(winter.index.tz_convert("Europe/London").hour).mean()
        s_loc = summer.groupby(summer.index.tz_convert("Europe/London").hour).mean()

        summer_winter_profiles.append({
            "house": h,
            "winter_utc_peak_hr": w_utc.idxmax(),
            "winter_utc_peak_w": round(w_utc.max(), 1),
            "summer_utc_peak_hr": s_utc.idxmax(),
            "summer_utc_peak_w": round(s_utc.max(), 1),
            "winter_loc_peak_hr": w_loc.idxmax(),
            "summer_loc_peak_hr": s_loc.idxmax(),
        })

    prof_df = pd.DataFrame(summer_winter_profiles)
    print(prof_df.to_string(index=False))
    print("\nSummary of Peak Hours across houses:")
    print("Winter UTC Peak Hour mode:", prof_df["winter_utc_peak_hr"].mode().tolist())
    print("Summer UTC Peak Hour mode:", prof_df["summer_utc_peak_hr"].mode().tolist())
    print("Winter Local Peak Hour mode:", prof_df["winter_loc_peak_hr"].mode().tolist())
    print("Summer Local Peak Hour mode:", prof_df["summer_loc_peak_hr"].mode().tolist())

if __name__ == "__main__":
    main()
