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

| Data Quality Check | Observed Value | Description & Engineering Action |
|---|---|---|
| **Timestamp Formatting** | 0 unparseable rows | Timestamps in `Time` and `Unix` parse cleanly with 100% valid records. |
| **Time vs Unix Timezone** | 0.0 h diff (Clean) / BST offset (Raw Part 1) | In the clean data, `Time - Unix_UTC == 0.0 h`. In raw Part 1, timestamps initially recorded UK local clock (BST = UTC+1). Official clean data adjusted timestamps by 1 h to true UTC. |
| **Ordering & Monotonicity** | 2 out-of-order steps | Sorting by Unix timestamp restores strict chronological ordering. |
| **Timestamp Duplicates** | 1,211,048 duplicate rows | Duplicates occur mainly in the Part 1 / Part 2 overlap window (June 27 to October 1, 2014) and during the autumn DST transition. 276,310 duplicate timestamps carry differing sensor values. Averaging duplicated timestamps preserves the energy state without arbitrary sample drop. |
| **Sampling Dropouts** | 4,040 intervals > 60 s | High-frequency polling dropouts. Intervals <= 5 minutes are linearly interpolated after 1-min resampling. |
| **Extended Outages** | 390 gaps > 5 min | Major gaps are tagged in the `outage` boolean mask and left unpopulated (NaN). |
| **Maximum Outage** | 998.45 hours (41.6 days) | February 2014 REFIT nationwide data acquisition system failure (2014-01-24 to 2014-03-06). |
| **Part 1 All-Zero Rows** | 0 rows | Part 1 used 0 for sensor disconnection; Part 2 used NaN. |
| **Negative Power Values** | 0 rows | Physically impossible active loads (< 0 W) were audited; 0 found in House 1. |
| **Appliance Spikes (> 4000 W)** | 3,949 spikes | Sensor transmission corruptions above the official REFIT physical ceiling (4 kW). Set to NaN to avoid corrupting energy sums. |
| **Aggregate Extremes (> 23 kW)** | 77 rows | Readings exceeding the domestic 100 A x 230 V UK fuse rating. Set to NaN. |
| **Sub-meter Sum Mismatch** | 50,228 rows | Instances where an individual appliance channel exceeds aggregate power due to sensor asynchrony (sensors polled sequentially over 6-8 s). |

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
2. **Deterministic Deduplication:** Group identical Unix timestamps and compute the mean across all channels. *Reason:* Simultaneous sensor reads describe the state at the exact same second; averaging avoids arbitrary record deletion while handling Part 1/Part 2 overlap.
3. **Physical Bounds & Anomaly Masking:**
   - Any appliance channel reading exceeding `APPLIANCE_SPIKE_W = 4000 W` is set to `NaN`. (*Reason:* The official REFIT threshold for corrupt transmissions. The official pipeline sets these to 0; we set them to NaN so gap and outage detection remains mathematically honest).
   - Any aggregate reading exceeding `AGG_EXTREME_W = 23000 W` is set to `NaN`. (*Reason:* UK domestic 100 A @ 230 V physical supply boundary).
   - Any negative active power reading is set to `NaN`.
4. **Regular 1-Minute Resampling:** Resample time series to regular 1-minute bins using `.mean()` for power channels. Power is strictly averaged, never summed.
5. **Gap Treatment:**
   - Missing intervals of duration `<= SHORT_GAP_MIN = 5 minutes` are linearly interpolated and tagged with `interpolated = True`. (*Reason:* Sensor packet drops during steady appliance runs can be faithfully reconstructed over short intervals).
   - Missing intervals of duration `> 5 minutes` are flagged in the `outage = True` boolean mask and preserved as `NaN`. (*Reason:* Filling multi-hour outages creates fictitious energy consumption).
6. **Artifact Output:** Serialized to `data/processed/house1_raw_clean_1min.parquet` with snappy compression and full boolean quality masks.

---

### 1.4 Diagnostic Visualizations

1. **Before vs After Cleaning (`figures/s1_cleaning_before_after.png`):**
   Demonstrates raw 8 s data versus cleaned 1-minute power across a 48-hour period (2013-10-26 to 2013-10-27 UTC) containing an active washing machine cycle (~2.4 kW heating pulses), cyclic fridge compressor activity, and peak aggregate loads. Sampling jitter is eliminated while preserving peak power and cycle energy.

2. **Gap Zoom & Outage Masking (`figures/s1_gap_zoom.png`):**
   Illustrates raw sensor dropouts on 2013-10-14, demonstrating linear interpolation of short gaps (<= 5 min) and unpopulated NaN masking of extended outages (> 5 min).

3. **Sampling Interval Distribution (`figures/s1_sampling_intervals.png`):**
   Validates nominal polling dynamics, showing a narrow distribution centered at 7-8 seconds with a heavy tail of network dropouts.
