# DATA.md: Dataset Architecture, Section 1 Raw Inspection & Data Quality Audit

This document records the data engineering findings, cleaning pipelines, threshold decisions, and audit tables for the REFIT Electrical Load Measurements dataset. Every number reported below is derived directly from the code outputs in `outputs/`.

---

## 1. Section 1: Raw House 1 Inspection and Cleaning Pipeline

### 1.1 Dataset Files and Scope
- **Files Used:** `RAW_House1_Part1.csv, RAW_House1_Part2.csv`
- **Date Coverage:** `2013-10-09 14:06:17+00:00` to `2015-07-10 11:56:32+00:00`
- **Nominal Sampling Interval:** 8 seconds (median observed: `7.0` s, 95th percentile: `13.0` s, 99th percentile: `15.0` s)
- **Measurement Units:** Active power in Watts (W) across all channels.
- **Monitored Channels:** Aggregate household power plus 9 individual sub-metered appliance channels:
  - Appliance 1: Fridge
  - Appliance 2: Chest Freezer
  - Appliance 3: Upright Freezer
  - Appliance 4: Washer Dryer
  - Appliance 5: Washing Machine
  - Appliance 6: Dishwasher
  - Appliance 7: Television
  - Appliance 8: Microwave
  - Appliance 9: Standby / unknown load

---

### 1.2 Inspection Findings and Anomaly Audit

The raw House 1 dataset consists of two chronological segments: `RAW_House1_Part1.csv` (4,377,877 rows) and `RAW_House1_Part2.csv` (4,140,886 rows), totaling 8,518,763 raw readings.

| Data Quality Check | Observed Value | Description & Empirical Observation |
|---|---|---|
| **Timestamp Formatting** | 0 unparseable rows | Timestamps in `Time` and `Unix` parse cleanly with 100% valid records. |
| **Time vs Unix Timezone** | 0.0 h diff (Clean) / 60 min offset (Raw Part 1 BST) | In `CLEAN_House1.csv`, `Time - Unix_UTC == 0.0 h`. In raw Part 1, monthly cross-correlation with cleaned data reveals a 60-minute offset during British Summer Time (Oct 2013, Apr–Sep 2014), where the raw logger recorded UTC+1. Official cleaned data shifted these periods by -60 min to true UTC. |
| **Ordering & Monotonicity** | 2 out-of-order steps | Sorting by Unix timestamp restores strict chronological ordering. |
| **Timestamp Duplicates** | 1,211,048 duplicate rows | 2,167,063 total rows fall into duplicated timestamp groups. 764,126 unique timestamps have duplicates within a single file, and 191,889 unique timestamps span both files during the overlap period (June 27 to October 1, 2014). 276,310 duplicate timestamps carry differing sensor values. Averaging duplicated timestamps provides a single value per second without discarding records. |
| **Sampling Dropouts** | 4,040 intervals > 60 s | Consecutive timestamp deltas greater than 60 seconds. Intervals <= 5 minutes are linearly interpolated after 1-min resampling. |
| **Extended Outages** | 390 intervals > 300 s (raw) / 375 gaps > 5 min (1-min grid) | Consecutive missing spans in the 1-minute grid exceeding 5 minutes. These are tagged in the `outage` boolean mask and left unpopulated (NaN). |
| **Maximum Outage** | 998.45 hours (41.6 days) | Extended recording gap from `2014-01-24 01:57:00` to `2014-03-06 16:22:00 UTC` (59,906 consecutive missing 1-minute intervals). |
| **Part 1 All-Zero Rows** | 0 rows | Rows where all 10 channels simultaneously equal 0.0 W. |
| **Negative Power Values** | 0 rows | Active power readings below 0 W. |
| **Appliance Spikes (> 4000 W)** | 3,949 spikes | Readings exceeding the `APPLIANCE_SPIKE_W = 4000 W` threshold. Set to NaN in the cleaning pipeline. |
| **Aggregate Extremes (> 23 kW)** | 77 rows | Whole-house aggregate readings exceeding the `AGG_EXTREME_W = 23000 W` threshold. Set to NaN. |
| **Sub-meter Exceeding Aggregate** | 50,228 rows | Instances where an individual sub-metered appliance channel records a value greater than the concurrent whole-house aggregate reading. |

#### Top 5 Largest Outages in House 1
| Outage Start (UTC) | Outage End (UTC) | Duration |
|---|---|---|
| 2014-01-24 01:57:00+00:00 | 2014-03-06 16:22:00+00:00 | 59,906 min (998.43 h) |
| 2015-05-10 23:37:00+00:00 | 2015-05-19 15:09:00+00:00 | 12,453 min (207.55 h) |
| 2015-06-10 13:20:00+00:00 | 2015-06-18 01:38:00+00:00 | 10,819 min (180.32 h) |
| 2015-04-03 01:44:00+00:00 | 2015-04-07 10:17:00+00:00 | 6,274 min (104.57 h) |
| 2015-03-22 08:16:00+00:00 | 2015-03-24 15:22:00+00:00 | 3,307 min (55.12 h) |

---

### 1.3 Cleaning Pipeline Rules

The cleaning pipeline (`src/cleaning.py`) processes raw House 1 data according to the following deterministic rules:

1. **Monotonic Sorting:** Concatenate Part 1 and Part 2 and sort strictly by Unix timestamp.
2. **Deterministic Deduplication:** Group identical Unix timestamps and compute the mean across all channels. Simultaneous sensor reads describe the power state at the same timestamp; averaging avoids arbitrary record deletion while resolving Part 1/Part 2 overlap.
3. **Physical Bounds & Anomaly Masking:**
   - Any appliance channel reading exceeding `APPLIANCE_SPIKE_W = 4000 W` is set to `NaN` (the threshold defined in official REFIT documentation; set to NaN rather than 0 so gap logic remains unskewed).
   - Any aggregate reading exceeding `AGG_EXTREME_W = 23000 W` is set to `NaN` (~100 A @ 230 V UK domestic supply ceiling).
   - Any negative active power reading is set to `NaN`.
4. **Regular 1-Minute Resampling:** Resample time series to regular 1-minute bins using `.mean()` for power channels. Power is strictly averaged, never summed.
5. **Gap Treatment:**
   - Missing intervals of duration `<= SHORT_GAP_MIN = 5 minutes` are linearly interpolated and tagged with `interpolated = True`.
   - Missing intervals of duration `> 5 minutes` are flagged in the `outage = True` boolean mask and preserved as `NaN`.
6. **Artifact Output:** Serialized to `data/processed/house1_raw_clean_1min.parquet` with snappy compression and full boolean quality masks.

---

### 1.4 Diagnostic Visualizations

1. **Before vs After Cleaning (`figures/s1_cleaning_before_after.png`):**
   Demonstrates raw 8 s data versus cleaned 1-minute power across a 48-hour period (2013-10-26 to 2013-10-27 UTC) containing an active washing machine cycle (~2.4 kW heating pulses), cyclic fridge compressor activity, and peak aggregate loads. Sampling jitter is eliminated while preserving peak power and cycle energy.

2. **Gap Zoom & Outage Masking (`figures/s1_gap_zoom.png`):**
   Illustrates raw sensor dropouts on 2013-10-14, demonstrating linear interpolation of short gaps (<= 5 min) and unpopulated NaN masking of extended outages (> 5 min).

3. **Sampling Interval Distribution (`figures/s1_sampling_intervals.png`):**
   Validates nominal polling dynamics, showing a narrow distribution centered at 7-8 seconds with a heavy tail of extended gaps.
