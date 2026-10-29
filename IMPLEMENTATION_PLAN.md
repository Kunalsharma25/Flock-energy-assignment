# REFIT Data Cleaning and Washing-Machine Disaggregation: Implementation Plan

Audience: a coding agent working on the user's local machine (likely Windows, so use `pathlib` and avoid shell-only tricks).
Goal: a public GitHub repo that satisfies the Flock Energy Data Science take-home assignment. Deadline: **11:00, 30 October 2026** (confirm the timezone with the user). Aim to submit at least 1-2 hours early.

---

## 0. Ground rules (read first, apply throughout)

1. **Simple and correct beats complex.** The reviewers say so explicitly. One LightGBM model with honest validation is the target. Do not add models, features or analyses that are not listed here until everything else is finished.
2. **Never invent numbers.** Every number in a `.md` doc must come from a file in `outputs/` produced by the code. Write `scripts/build_doc_tables.py` to turn the CSVs into markdown tables and paste from those. Interpretations of plots must describe what the plot actually shows.
3. **Every decision gets a stated rule and a one-line reason**, recorded in `src/config.py` and in the docs.
4. **Thresholds are provisional until frozen** (Phase 4). After freezing, do not change them to improve results. If a change is unavoidable, record it and why in `DATA.md`.
5. **No leakage.** The test houses are not touched until the final evaluation. No random row splits. No appliance channels as model inputs.
6. **Reproducible.** Fixed seeds (`SEED = 42`), pinned `requirements.txt`, relative paths only, one command per stage, and a clean-clone test at the end.
7. **STOP & REPORT points.** Where this document says `STOP & REPORT`, pause, print the requested outputs to the user, and wait for their go-ahead. Do not proceed on your own.
8. **If an assumption in Section 1 turns out wrong, say so and adapt.** Do not silently work around it.
9. **Keep code short, commented and explainable.** The user will be asked to defend every line in an interview.
10. **Do not open or edit the large CSVs in Excel.** Use Python only.

---

## 1. Facts and assumptions

### Verified from the dataset documentation and metadata workbook
- 20 households labelled House 1-21, **no House 14**. Each has 1 aggregate sensor plus 9 Individual Appliance Monitors (IAMs). Active power only, **watts**, nominal **8 s** sampling.
- Sensors are not synchronised (the collection script polled every 6-8 s). Values were recorded on load change and filled in between.
- Raw data is untouched apart from a UK daylight-saving correction. Raw file layout: `Datetime, Unix timestamp (UTC), Aggregate, Appliance1..Appliance9`.
- Raw file names containing `_Part1` have unavailable sensors set to **0**; `_Part2` set them to **NaN**.
- Cleaned data (official): NaNs forward-filled, IAM spikes **above 4000 W set to zero**, and an **`Issues` column = 1** when the sum of IAMs exceeds the aggregate (discard those rows).
- Known outages exist, notably **February 2014**.
- Coverage is described inconsistently (readmes say 2013-2014, the dataset page says Oct 2013 - Jun 2015, and metadata notes date equipment changes into Mar/Apr 2015). **Compute actual coverage from the files** and state it.
- The cleaned archive (`CLEAN_REFIT_081116.7z`) contains `CLEAN_House1.csv` ... `CLEAN_House21.csv` (no 14), each about 250-425 MB uncompressed, about 6.8 GB in total. **The user cannot extract everything at once**, so process one house at a time.

### Assumptions the agent must VERIFY in Phase 1-2 and report
- A1. Cleaned CSV header is `Time, Unix, Aggregate, Appliance1..Appliance9, Issues` with the Unix column in position 1. Verify by reading the header.
- A2. Whether the `Time` column is UTC or UK local time (compare `Time` with `Unix`, see Phase 2).
- A3. File names inside the raw archive (Part1/Part2?) and how many files House 1 has.
- A4. Whether each washing-machine channel really looks like a washing machine (Phase 4 signature check).
- A5. Whether the archive is solid (affects extraction speed only).

---

## 2. Repository layout

```
refit-nilm/
├── README.md  DATA.md  RESULTS.md  Recommendation.md
├── requirements.txt
├── .gitignore                 # data/, __pycache__/, .ipynb_checkpoints/
├── run_all.py                 # stages: download, prepare, section1, quality, eda, model, docs
├── Makefile                   # optional wrapper (Windows users run run_all.py)
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py        # per-house extract -> 1-min parquet -> delete CSV
│   └── build_doc_tables.py    # CSV -> markdown tables for the docs
├── src/
│   ├── __init__.py  config.py  data_io.py  cleaning.py  quality.py
│   ├── cycles.py  eda.py  splits.py  features.py  models.py  evaluate.py  plotting.py
├── notebooks/                 # thin: import src/, show figures, write interpretation
│   ├── 01_raw_cleaning.ipynb  02_wm_eda.ipynb  03_disaggregation.ipynb
├── tests/test_cycles.py  tests/test_resample.py
├── outputs/                   # generated tables/json (COMMITTED)
├── figures/                   # generated plots (COMMITTED)
└── data/                      # GITIGNORED: archives/ tmp/ raw/ processed/
```

`requirements.txt` (pin the versions that were actually installed): `pandas numpy pyarrow lightgbm scikit-learn matplotlib seaborn py7zr requests pytest jupyter nbconvert`.

---

## 3. `src/config.py`: use these values

```python
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("REFIT_DATA", ROOT / "data"))
ARCHIVE_DIR, TMP_DIR = DATA_DIR / "archives", DATA_DIR / "tmp"
RAW_DIR, PROC_DIR = DATA_DIR / "raw", DATA_DIR / "processed"
OUT_DIR, FIG_DIR = ROOT / "outputs", ROOT / "figures"
SEED = 42
TZ_LOCAL = "Europe/London"

ARCHIVES = {
    "raw":   {"file": "REFIT_RAW_081116.7z",
              "urls": ["https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z",
                       "https://pure.strath.ac.uk/ws/portalfiles/portal/62090177/REFIT_RAW_081116.7z"]},
    "clean": {"file": "CLEAN_REFIT_081116.7z",
              "urls": ["https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z",
                       "https://pure.strath.ac.uk/ws/portalfiles/portal/62090184/CLEAN_REFIT_081116.7z"]},
}
# The Strathclyde site may block scripted downloads. If so, tell the user to download the two
# archives manually into data/archives/ (Zenodo mirror: DOI 10.5281/zenodo.5063428, record in README).

# (house, appliance_index) of washing-machine channels. Source: MetaData_Tables.xlsx and READMEs.
# House 12 has no washing machine channel.
WM_CANDIDATES = [(1,5),(2,2),(3,6),(4,4),(4,5),(5,3),(6,2),(7,5),(8,4),(9,3),
                 (10,5),(11,3),(13,3),(15,3),(16,5),(17,4),(18,5),(19,2),(20,4),(21,3)]
TUMBLE_DRYER  = {3:4, 5:2, 7:4, 15:2, 17:3, 20:3, 21:2}
WASHER_DRYER  = {1:4, 8:3, 9:2, 18:4}
DISHWASHER    = {1:6, 2:3, 3:5, 5:4, 6:3, 7:6, 9:4, 10:6, 11:4, 13:4, 15:4, 16:6, 18:6, 20:5, 21:4}
H1_CHANNELS   = {"fridge": 1, "freezer1": 2, "washer_dryer": 4, "washing_machine": 5,
                 "dishwasher": 6}                      # for Section 1 plots
OCCUPANCY = {1:2,2:4,3:2,4:2,5:4,6:2,7:4,8:2,9:2,10:4,11:1,12:3,13:4,15:1,16:6,17:3,18:2,19:4,20:2,21:4}
APPLIANCES_OWNED = {1:35,2:15,3:27,4:33,5:44,6:49,7:25,8:35,9:24,10:31,11:25,12:26,13:28,
                    15:19,16:48,17:22,18:34,19:26,20:39,21:23}
H13_WM_END_UTC = "2015-03-25"   # H13 washing machine changes on this date: use data BEFORE it only
MODEL_EXCLUDE_HOUSES = {4}      # two machines; ambiguous target. Kept in EDA only (as 4a=ch4, 4b=ch5).

# --- PROVISIONAL thresholds (tuned once in Phase 4, then FROZEN) ---
APPLIANCE_SPIKE_W = 4000        # official REFIT rule; used on raw appliance channels
AGG_EXTREME_W     = 23000       # ~100 A x 230 V UK supply limit; flag above this (check distribution)
SHORT_GAP_MIN     = 5           # interpolate gaps <= this (Section 1)
STALE_MIN         = 30          # cleaned data: all channels unchanged for >= this many minutes = stale
ON_W              = 20.0        # washing machine ON above this power
MERGE_GAP_MIN     = 10          # merge ON segments separated by <= this many minutes (mid-cycle pauses)
MIN_CYCLE_MIN     = 10          # discard cycles shorter than this
MIN_CYCLE_WH      = 20.0        # discard cycles with less energy than this
# eligibility (Phase 4)
MIN_USABLE_DAYS   = 90
MIN_CYCLES        = 100
SIG_DUR_RANGE_MIN = (20, 240)   # median cycle duration plausible range
SIG_MIN_PEAK_W    = 500         # median cycle peak power at least this
```

Everything downstream imports from here. No magic numbers elsewhere.

---

## 4. Phases

### Phase 0: Setup (about 20 min)
1. Create the repo layout above, `git init`, `.gitignore`, virtual environment, install requirements, `pytest` runs (even with no tests).
2. Check free disk space and report it. Need about 2 GB free (archive sizes + one extracted CSV at a time).
3. Check for the 7-Zip CLI (`7z` on PATH, or `C:\Program Files\7-Zip\7z.exe`). If absent, use `py7zr`.

**Done when:** `python -c "import src.config"` works and `pytest` exits cleanly.

### Phase 1: Data acquisition (about 30 min, mostly waiting)
1. `scripts/download_data.py`: download each archive into `data/archives/` (skip if present, stream to a `.part` file then rename, try each URL, send a browser-like User-Agent). If every URL fails, print clear manual-download instructions and exit non-zero.
2. **List archive contents without extracting** (`7z l -slt <archive>`) and save the listing to `outputs/archive_listing_{raw,clean}.txt`. Record: file names, uncompressed sizes, whether the archive is Solid (assumption A5), and the raw House 1 file names (assumption A3).
3. Extract **only the raw House 1 file(s)** into `data/raw/`.

**STOP & REPORT:** show the listing summary, the House 1 raw file names, and the first 3 lines of one raw House 1 file and one cleaned file (stream a cleaned file with `7z e -so`, or extract House 1 once). Confirm assumptions A1-A3 and wait for approval before Phase 2.

### Phase 2: Prepare 1-minute parquet files for the cleaned data (about 20-40 min compute)
`scripts/prepare_data.py`, with a `--houses` option (default: all 19 houses that have a file, including House 12 for completeness).

For each house, one at a time:
1. Extract `CLEAN_House{N}.csv` to `data/tmp/` (7z CLI, or `py7zr` with `targets=[...]`). If the archive stores files in a sub-folder, handle the path.
2. Read with `usecols` by **header name** (Unix, Aggregate, Appliance1-9, Issues), downcast to float32. If memory is a problem, read in chunks and accumulate per-minute `sum` and `count`, then combine.
3. Drop rows with missing Unix. Sort by Unix. Index = `pd.to_datetime(unix, unit="s", utc=True)`.
4. Resample to 1 min: **mean** for power channels, **max** for `Issues`. Do not sum power.
5. Rename columns to `aggregate, a1..a9, issues`. Save `data/processed/house_{N}_1min.parquet` (UTC-indexed).
6. Delete the temp CSV. Print rows, first/last timestamp and elapsed time.

Also, for House 1 only, test assumption A2: read the `Time` and `Unix` columns, compute `(Time - Unix_as_UTC)` in hours, and print `value_counts()`. Interpretation: `0` only means `Time` is UTC; `0` and `1` means `Time` is UK local time with BST. Save the result to `outputs/tz_check.txt`.

**Done when:** one parquet per house exists, `data/tmp/` is empty, and `tests/test_resample.py` passes (a constant 100 W input resamples to 100 W, never 100 x samples).

### Phase 3: Section 1, raw House 1 inspection and cleaning (about 2-3 h)
Code in `src/cleaning.py`, run by `run_all.py --stage section1`. Use the **raw** House 1 files.

**3a. Load.** Concatenate all House 1 raw files, index by the Unix column (UTC).

**3b. Inspection.** Compute and write `outputs/house1_quality_summary.csv` (and later copy into `DATA.md`):
- Row count, first/last timestamp, files used, Part1/Part2 status.
- Timestamp parsing: any unparseable rows; `Time` vs `Unix` agreement (use the result of the tz check).
- Ordering: count of out-of-order rows (`diff() < 0`); sort afterwards.
- Duplicates: count of duplicated Unix timestamps; of those, how many carry *different* values.
- Irregular intervals: distribution of consecutive deltas (median, 95th/99th percentile, max), count of deltas above 60 s and above 5 min, table of the 10 largest gaps with start/end.
- Missing values: NaNs per channel. In Part1 files also count rows where all 10 channels are exactly 0 (treated as sensor-unavailable; justify: a house aggregate of exactly 0 W with every appliance at 0 is implausible).
- Impossible/extreme values: negatives per channel; appliance channels above 4000 W; aggregate above `AGG_EXTREME_W`; rows where one appliance exceeds the aggregate; flat-lined runs (constant value for a long time) per channel.
- DST check: look around the 4 transitions (27 Oct 2013, 30 Mar 2014, 26 Oct 2014, 29 Mar 2015) for duplicated or reordered timestamps.

**3c. Cleaning pipeline (document each rule as a docstring and in `DATA.md`):**
1. Sort by Unix.
2. **Duplicates:** average duplicated timestamps (state why: both readings describe the same instant; report how many differed).
3. Set to NaN: appliance values above `APPLIANCE_SPIKE_W`, negatives, aggregate above `AGG_EXTREME_W`, and Part1 all-zero rows. (Note in `DATA.md` that the official pipeline sets appliance spikes to zero, whereas we use NaN so the gap logic stays honest.)
4. **Resample to 1 min with the mean.**
5. **Gaps:** linearly interpolate gaps of `SHORT_GAP_MIN` minutes or fewer. Leave longer gaps as NaN and add boolean columns `interpolated` and `outage` (long gap). Do not fill outages.
6. Output `data/processed/house1_raw_clean_1min.parquet` and a table of outages (start, end, duration).

**3d. Plots** (`figures/s1_*.png`): aggregate plus **fridge (ch 1) and washing machine (ch 5)** (add dishwasher if informative), **before vs after** cleaning over a chosen 1-2 day window. Choose the window by a rule stated in the caption (for example the first 2-day window containing a washing-machine cycle and at least one flagged value). Add one zoom on a gap showing interpolated vs flagged-outage minutes. Add a one-line caption stating what changed.

**3e. Write the `DATA.md` Section 1 content**: files used, date coverage, sampling interval, units, findings table, rules, remaining concerns.

**Done when:** the summary CSV, the cleaned parquet, 3+ figures and the draft `DATA.md` section exist, and re-running the stage reproduces them exactly.

**STOP & REPORT:** show the findings table and the figures. Wait for approval.

### Phase 4: Quality masks, cycle rule, availability table, house selection, freeze (about 2 h)
`src/quality.py`, `src/cycles.py`.

**4a. Usable-minute mask (cleaned data).** For each house parquet:
- `present` = aggregate is not NaN.
- `stale` = minutes inside a run of at least `STALE_MIN` consecutive minutes in which **all** channels (aggregate and a1..a9) changed by less than 0.01 W. These are forward-fill artefacts. Before fixing `STALE_MIN`, look at the histogram of run lengths and at a known outage (Feb 2014) in at least one house, and save the evidence as `figures/s2_stale_check.png`.
- `bad_issue` = `issues == 1`.
- `usable = present & ~stale & ~bad_issue`.

**4b. Cycle detector** (`src/cycles.py`), applied to the WM power series with unusable minutes set to NaN:

```python
import numpy as np, pandas as pd
from . import config as C

def long_runs(mask, min_len):
    m = np.asarray(mask, bool); out = np.zeros_like(m)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= min_len: out[s:e] = True
    return out

def run_lengths(mask):
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)

def detect_cycles(p, on_w=C.ON_W, merge_gap=C.MERGE_GAP_MIN, min_dur=C.MIN_CYCLE_MIN, min_wh=C.MIN_CYCLE_WH):
    """p: 1-min mean watts, regular DatetimeIndex, NaN = unusable. NaN counts as OFF."""
    on = (p > on_w).fillna(False).to_numpy()
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    runs = [[s, e] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_gap: merged[-1][1] = e
        else: merged.append([s, e])
    v, nan = p.fillna(0).to_numpy(), p.isna().to_numpy()
    rows = []
    for s, e in merged:
        seg = v[s:e]; wh = seg.sum() / 60.0
        touches_gap = nan[max(s-1, 0)] or nan[min(e, len(nan)-1)] or nan[s:e].any()
        if (e - s) >= min_dur and wh >= min_wh and not touches_gap:
            rows.append(dict(start=p.index[s], end=p.index[e-1], duration_min=e-s,
                             energy_wh=wh, peak_w=seg.max(), mean_w=seg.mean()))
    return pd.DataFrame(rows)
```
Cycles that touch an unusable minute are discarded (a cycle cut by an outage would be mis-measured). `tests/test_cycles.py` must cover: a pause shorter than the merge gap is merged, a short blip is dropped, a cycle next to a NaN is discarded, and energy for a constant 1000 W over 60 min is 1000 Wh.

**4c. Tune the cycle rule ONCE.** Plot at least 20 randomly selected detected cycles from at least 3 houses (`figures/s2_cycle_tuning.png`), judge whether the rule captures whole cycles without splitting or merging unrelated ones, adjust `ON_W`, `MERGE_GAP_MIN`, `MIN_CYCLE_MIN`, `MIN_CYCLE_WH` if clearly needed, and record the before/after values and reason.

**4d. Availability table.** For every candidate in `WM_CANDIDATES`, writing `outputs/availability_clean.csv`: house, channel, first/last usable timestamp, usable days, % present, % stale, % issues==1, longest outage (min), number of outages of 60+ min, % zero, % off (`<= ON_W`) of usable time, n cycles, median duration/energy/peak, cycles per week (cycles divided by usable weeks), occupancy, appliances owned, has tumble dryer, has washer-dryer, notes. For House 13, truncate to before `H13_WM_END_UTC`. House 4 appears twice (`4a` = ch4, `4b` = ch5).

**4e. Eligibility filters.** A candidate is eligible if all hold:
- usable days >= `MIN_USABLE_DAYS`,
- n cycles >= `MIN_CYCLES`,
- median cycle duration in `SIG_DUR_RANGE_MIN` and median peak >= `SIG_MIN_PEAK_W` (plausible washing-machine signature; this catches mislabelled channels, check H8, H13 and H4 in particular).

Write `outputs/eligibility.csv` with a pass/fail per filter and a reason for each exclusion. If fewer than 6 candidates pass, relax `MIN_CYCLES` or `MIN_USABLE_DAYS`, and say so. The brief requires at least 4 for EDA.

**4f. Freeze.** Write `outputs/selection_frozen.json` with all thresholds, the eligible candidates and the date. `eda`, `splits` and `models` read this file; nothing downstream edits it.

**STOP & REPORT:** show the availability table, the eligibility result, the tuning figure and the proposed frozen thresholds. Wait for the user's approval before freezing. This is the key review point.

### Phase 5: Section 2, washing-machine EDA (about 2-3 h)
`src/eda.py`, notebook `02_wm_eda.ipynb`. Use **all eligible candidates** (descriptive, so no leakage risk). All hour/weekday analysis uses `TZ_LOCAL` (`index.tz_convert("Europe/London")`), after the DST check from Phase 2.

Compute and save to `outputs/`:
- `cycle_summary.csv`: per house, usable period, usable days, % missing, % stale, % off, % zero, n cycles, and the cycle characteristics (duration, energy, peak power, mean power).
- `usage_by_hour.csv` and `usage_by_weekday.csv`: cycle starts per hour and per weekday, as raw counts and normalised share per house.

Required figures (`figures/s2_*.png`), each followed by a **2-3 sentence interpretation written from the actual numbers**:
1. **Representative 24-hour trace**: aggregate plus washing-machine power for one day. Selection rule (state it in the caption): among days with 100% usable coverage and at least 2 cycles in the median-cycle-count house, pick the day with median WM energy.
2. **Detected cycles**: 3-4 cycles overlaid or in small panels with detected start/end marked, power on the y-axis.
3. **Cross-house comparison**: boxplot of energy per cycle by house (and duration as a second panel).
Additional supporting figures: hour-of-day and day-of-week profiles (per house, normalised, plus a pooled hour x weekday heatmap), and cycles per week vs occupancy (describe as an observation, since n is small).

Write the plain-language cycle-rule explanation in `RESULTS.md`. State which houses and periods were used and which candidates were excluded and why.

**Done when:** all tables and figures exist and the interpretations in `RESULTS.md` match them.

### Phase 6: Section 3, disaggregation model (about 3 h)
`src/splits.py`, `features.py`, `models.py`, `evaluate.py`; notebook `03_disaggregation.ipynb`.

**6a. Data.** Eligible houses except `MODEL_EXCLUDE_HOUSES`. For each house build a regular 1-min frame, set `aggregate` and the WM target to NaN wherever `usable` is false, **then** build features. Drop any row with a NaN feature or target. A window that touches an unusable minute is therefore dropped. Target = WM power in watts (clip at >= 0).

**6b. Split by whole houses (primary design).** Strata: `confounder` (has a tumble-dryer or washer-dryer channel) and `no_confounder`. With a seeded `numpy.random.default_rng(SEED)` over sorted house lists:
- **Test:** 2 houses, one per stratum where possible, preferring different occupancy.
- **Validation:** 1-2 houses (one per stratum if possible).
- **Train:** all remaining eligible houses (need at least 3; if fewer, switch to leave-one-house-out cross-validation and report per-fold results).
- Write `outputs/model_split.json` listing houses and date ranges per role.
- Fit nothing on validation or test data (including scalers and thresholds); the test houses are untouched until the final evaluation.

Explain in `README.md` why this avoids leakage: houses are disjoint across roles, so no overlapping or adjacent minutes cross a boundary, and a random row split would put near-identical neighbouring minutes in both train and test.

*Optional, only if time remains:* a secondary "seen-home" experiment (chronological split inside each home, with a buffer of at least the feature window), clearly labelled as a different scenario.

**6c. Features (from the aggregate only; the time-based ones come from the timestamp):**
```python
OFFSETS = [-30,-20,-15,-10,-5,-3,-2,-1,0,1,2,3,5,10,15,20,30]   # minutes; future samples are legitimate offline
def build_features(df):
    a = df["aggregate"]                       # regular 1-min index, NaN where unusable
    X = {f"agg_t{k:+d}": a.shift(-k) for k in OFFSETS}
    for w in (5, 15, 31):
        r = a.rolling(w, center=True, min_periods=w)
        X.update({f"mean_{w}": r.mean(), f"std_{w}": r.std(), f"min_{w}": r.min(), f"max_{w}": r.max()})
    X["d_prev"] = a - a.shift(1); X["d_next"] = a.shift(-1) - a
    X["above_base60"] = a - a.rolling(61, center=True, min_periods=61).min()
    loc = a.index.tz_convert(C.TZ_LOCAL)
    h = loc.hour + loc.minute / 60
    X["hour_sin"], X["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    X["dow_sin"], X["dow_cos"] = np.sin(2*np.pi*loc.dayofweek/7), np.cos(2*np.pi*loc.dayofweek/7)
    return pd.DataFrame(X, index=a.index)
```
State in `README.md` that using future samples is intentional (offline disaggregation, as in Seq2Point's centred window).

**6d. Models:**
| Name | Description |
|---|---|
| Baseline 1 | Always off (predict 0) |
| Baseline 2 | Mean WM power by (weekday, hour) computed on the training houses only, applied to test houses |
| Main model | `lightgbm.LGBMRegressor`, objective `regression`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=100`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=2000` with early stopping (50 rounds) on the **validation houses**, `random_state=SEED` |

Post-processing: clip predictions to `[0, aggregate]` (the washing machine cannot draw more than the household). Choose the ON-state decision threshold for predictions from a small grid (10, 20, 30, 50, 75, 100 W) by maximising F1 on the **validation** houses, then freeze it for the test.
Stretch goal, only after everything else is done: a small Seq2Point-style 1D CNN as an extra row in the table.

**6e. Metrics** (`src/evaluate.py`), reported **per test house and pooled**, always **next to both baselines** in `outputs/model_metrics.csv`:
- **MAE and RMSE** (W) on all usable minutes, plus MAE on true-ON minutes and true-OFF minutes separately (an always-off baseline looks excellent on plain MAE because the machine is off most of the time; say so).
- **SAE** = |sum(pred) - sum(true)| / sum(true): total-energy error, the most utility-relevant metric.
- **ON-state precision, recall, F1** (true ON = power above `ON_W`; predicted ON = prediction above the validated threshold).
- **Cycle-level:** run `detect_cycles` on the predicted series. A true cycle is *detected* if predicted ON time covers at least 50% of its duration; cycle recall = detected/true; cycle precision = predicted cycles overlapping at least 50% of their own duration with a true cycle / predicted cycles; median absolute % error in energy per matched cycle.

Justify each metric in one line in `RESULTS.md`.

**6f. Figures** (`figures/s3_*.png`):
1. **Actual vs predicted** for a test house: top panel aggregate, lower panel actual vs both baselines vs the model, over one day. Selection rule stated in the caption (for example: the test-house day with full usable coverage and the median number of cycles among days with at least 2 cycles).
2. **Failure case 1 (confounding load):** the test-house day where the dryer/dishwasher channel overlaps a true washing cycle with the largest model error. Overlay the dryer or dishwasher channel (diagnostic only, never a feature).
3. **Failure case 2:** the largest-error day overall, or a house whose cycle signature (peak/duration from `cycle_summary.csv`) differs most from the training houses, explaining why it transfers poorly.

Add 2-3 sentences per figure on what went wrong and when the model is likely to struggle.

**6g.** Run the model-selection code only on train/validation. Evaluate on test houses exactly once. Save predictions to `outputs/test_predictions.parquet`.

**STOP & REPORT:** show `model_split.json`, the metrics table, and the three figures. Wait for approval before writing the final docs.

### Phase 7: Section 4 and documentation (about 2 h)
Write all docs from the generated outputs (use `scripts/build_doc_tables.py` for tables).

**`DATA.md`:** files and houses used, actual date coverage per house (from `availability_clean.csv`), sampling interval (8 s nominal, 1 min after resampling), units (W), Section 1 findings and rules, the cleaned-data issues (forward-fill, 4000 W zeros, Issues flag, stale detection) and how each was handled, remaining concerns (unresolved coverage discrepancy, unmonitored loads, sensor asynchrony, channel verification results).

**`RESULTS.md`:** the washing-machine EDA with figures and interpretations, the cycle rule in plain language, model split, metrics table, validation reasoning, failure analysis, and a short honest summary of what the model can and cannot do.

**`Recommendation.md`** (utility-facing; the summary page first, about 400-500 words, then the Section 4 answers):
- Suitable uses: cohort-level insight into when washing happens, total WM energy estimates (SAE), load-shifting targeting. Unsuitable: billing-grade, per-cycle measurement for an individual home, homes unlike the training set.
- **One opportunity:** shifting washing-machine starts out of a stated peak window. Compute from the EDA the share of cycle starts inside the window, and state the window as an assumption (not as a UK tariff fact). Measurement plan: pre/post or randomised nudge trial with a control group, primary metric = share of cycles starting inside the target window, secondary = energy shifted, with seasonality controls and a pre-registered minimum effect size.
- **Limitations:** overlapping appliance signatures, missing/flagged data, household differences (occupancy, machine type), unmonitored loads, and **UK vs India** (top-loaders and semi-automatic machines, cold-water washing so little heating-element signature, voltage fluctuation, different appliance mix, tariff and usage culture). State that the model has not been validated on Indian homes.
- **15/30-minute data paragraph:** a 1-2 hour cycle becomes about 2-8 samples, cycle edges and power peaks are smeared, short loads become undetectable, event/step-based features disappear, and you would move from cycle-level disaggregation to usage-pattern or energy-share inference.

**`README.md`:** what was built; setup and run instructions (`pip install -r requirements.txt`, `python run_all.py --stage all`, estimated runtime, disk need); assumptions; preprocessing decisions; validation design; trade-offs (why LightGBM, why house-level split, why 1-min, why H4 is excluded from modelling); **AI tools used** (ask the user to confirm exact wording; suggested: "Claude (Anthropic) was used for planning and review; the Antigravity coding agent was used for implementation. The author reviewed, ran and understands all code and results."); possible improvements (Seq2Point, more appliances, transfer learning, quantile or event-based models, leave-one-house-out CV, India-specific data); references.

**References:** cite only papers you have verified exist; search for each one and drop any you cannot confirm. Candidates (verify authors, year, venue):
- Murray, Stankovic, Stankovic (2017), *Scientific Data*, doi 10.1038/sdata.2016.122 (REFIT dataset).
- Murray et al. (2015), "A data management platform for personalised real-time energy feedback", EEDAL (cited in the readme).
- Kelly & Knottenbelt (2015), "Neural NILM", BuildSys.
- Zhang et al. (2018), "Sequence-to-point learning with neural networks for non-intrusive load monitoring", AAAI.
- Batra et al. (2014), NILMTK, e-Energy.
- Ke et al. (2017), LightGBM, NeurIPS.
- One NILM metrics paper (for example Makonin & Popowich 2015, or a Klemenjak et al. metrics paper) to justify SAE and the F1 choices.

For each citation write one line on what idea was **adopted** (for example the centred window from Seq2Point, SAE from the metrics literature).

### Phase 8: Reproducibility check (about 1-2 h, do not skip)
1. Execute notebooks top to bottom (`jupyter nbconvert --execute --inplace`) so no stale outputs remain.
2. `pytest` passes.
3. Clone the repo into a **fresh folder**, create a new virtual environment, install requirements, run `python run_all.py --stage all` (with the archives already in `data/archives/` or downloaded), and confirm the outputs and figures regenerate with matching numbers. Fix anything that breaks (hard-coded paths, missing imports, undocumented steps).
4. Check that the docs contain no numbers that disagree with `outputs/`.

### Phase 9: Submission checklist
- [ ] Public GitHub repo; link opens without login. Confirm with the user how the company wants the link submitted (email address, form, and format).
- [ ] `README.md`, `DATA.md`, `RESULTS.md`, `Recommendation.md` present at repo root.
- [ ] Source code runs end to end from supplied data through cleaning, EDA, predictions and evaluation.
- [ ] `outputs/` and `figures/` committed; `data/` not committed; repo size is small.
- [ ] Every threshold and rule is stated in plain language in the docs.
- [ ] AI-tool use disclosed truthfully.
- [ ] Citations verified.
- [ ] A short "what I would do next and why" section exists in the README for anything left undone.
- [ ] Submitted at least 1-2 hours before 11:00 on 30 Oct 2026.

---

## 5. If time runs short: priority order
1. Section 1 complete with plots and `DATA.md` (Phase 3).
2. Availability table, eligibility and frozen thresholds (Phase 4).
3. Section 2 required three plots (Phase 5).
4. Baselines plus LightGBM with the house-level split and the metrics table (Phase 6).
5. The four docs and the clean-clone test (Phases 7-8).
6. Extras (Seq2Point, leave-one-house-out CV, seen-home experiment): only after the above is done and committed.

If something is not finished, say so plainly in the README, with what you would do next and why. The brief rewards that over incomplete or unverifiable work.

## 6. Common mistakes to avoid
- Random train/test split on time series (leakage).
- Judging only by MAE or accuracy while an always-off model looks good.
- Summing instead of averaging when resampling power.
- Silently filling long gaps, or treating forward-filled stale stretches in the cleaned data as real readings.
- Training or evaluating on rows with `Issues == 1`.
- Choosing houses or thresholds by looking at test performance.
- Using appliance channels (dryer, dishwasher) as model inputs.
- Plots without interpretation text, or unstated thresholds.
- Notebooks with stale outputs; absolute paths; unpinned requirements.
- Quoting numbers in docs that were typed by hand instead of taken from `outputs/`.
