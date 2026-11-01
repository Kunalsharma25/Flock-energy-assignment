# REFIT Washing Machine NILM Disaggregation

> **Flock Energy Data Science Take-Home Assignment**  
> Dataset: [REFIT Electrical Load Measurements](https://doi.org/10.1038/sdata.2016.122) 

---

## What Was Built

A reproducible, end-to-end pipeline that:

1. **Audits and cleans** the raw House 1 REFIT data, documenting every decision rule.
2. **Resamples** all 20 cleaned houses to regular 1-minute Parquet files.
3. **Detects and characterises** domestic washing-machine cycles across 20 homes (19/20 eligible), with a detailed exploratory data analysis (EDA) on four selected cohort homes.
4. **Trains and evaluates** a LightGBM NILM disaggregation model using strict whole-house splits (no appliance channels as inputs; no random row splits) to estimate washing-machine power from aggregate-only smart meter readings.
5. **Documents** findings, failure modes, and business recommendations in four Markdown files.

---

## Repository Layout

```
├── README.md               # This file
├── DATA.md                 # Dataset architecture, quality audit, and EDA findings
├── RESULTS.md              # Modelling methodology, metrics, and failure analysis
├── Recommendation.md       # Utility-facing summary and Section 4 answers
├── requirements.txt
├── run_all.py              # Master pipeline entry point
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── run_disaggregation.py   # Phase 6 model pipeline (main entry)
│   ├── finalize_metrics.py
│   ├── regen_s3_figures.py
│   └── run_loho_cv.py
├── src/
│   ├── config.py           # All thresholds and constants (single source of truth)
│   ├── cleaning.py         # Raw House 1 cleaning pipeline
│   ├── quality.py          # Phase 4 quality masks and eligibility
│   ├── cycles.py           # Cycle detection, flatline masking, physical ceilings
│   ├── eda.py              # Section 2 EDA
│   ├── splits.py           # Whole-house train/val/test splits
│   ├── features.py         # 36 aggregate-only features on 1-min grid
│   ├── models.py           # Baselines 1–3 and LightGBM regressor
│   └── evaluate.py         # Metrics and diagnostic figures
├── tests/
│   ├── test_config.py
│   ├── test_cycles.py
│   ├── test_models.py
│   └── test_resample.py
├── outputs/                # Generated tables and JSON (committed)
└── figures/                # Generated plots (committed)
```

`data/` is gitignored (archives, tmp, raw, processed).

---

## Setup and Replication

### Requirements
- Python 3.10+
- ~2.5 GB free disk space (archives + one extracted CSV at a time)
- The two REFIT archives placed in `data/archives/`:
  - `REFIT_RAW_081116.7z`
  - `CLEAN_REFIT_081116.7z`

Both archives are available at:
- Strathclyde Pure portal: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
- Zenodo mirror: https://doi.org/10.5281/zenodo.5063428

### Install Dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_all.py --stage all
```

| Stage | Command | Description | Approx. Runtime |
|:---|:---|:---|:---|
| `download` | `--stage download` | Downloads archives (skip if already present) | Network-dependent |
| `prepare` | `--stage prepare` | Resamples all 20 cleaned houses to 1-min Parquets | ~10–20 min |
| `section1` | `--stage section1` | Raw House 1 audit, cleaning, figures, DATA.md Section 1 | ~5 min |
| `quality` | `--stage quality` | Quality masks, stale tuning, cycle detection, eligibility freeze | ~10 min |
| `eda` | `--stage eda` | Section 2 EDA on cohort homes, cycle summary, figures | ~5 min |
| `model` | `--stage model` | LightGBM training (17 houses, 10.8M rows), evaluation, figures | ~20–40 min |

### Run Tests

```bash
pytest tests -v
```

All 14 tests pass.

---

## Key Assumptions and Preprocessing Decisions

### Data Assumptions (All Verified in Phase 1–2)
- **Channel layout:** `Time, Unix, Aggregate, Appliance1..Appliance9, Issues` (UTC timestamps).
- **Raw Part 1 BST offset:** `RAW_House1_Part1.csv` logged wall-clock time during BST periods (Oct 2013, Apr–Sep 2014), offset by +60 min relative to UTC. The official cleaned dataset corrects this. Raw Part 1 and Part 2 overlap from 2014-06-27 to 2014-10-01; 276,310 duplicate timestamps carry differing values and are resolved by averaging.
- **`Issues == 1` flag:** Rows where any appliance channel sum exceeds the aggregate (REFIT checksum). These rows are excluded from model features and targets but *not* used to gate cycle detection — doing so would discard 91–97% of authentic wash cycles (see `DATA.md` Section 2.1).
- **No House 14** exists in either archive.

### Cleaning Rules (Deterministic, No Manual Intervention)
1. Sort by Unix timestamp; average duplicate timestamps.
2. Set appliance readings > 4,000 W to NaN; aggregate readings > 23,000 W to NaN; negative readings to NaN.
3. Resample to 1-minute bins using `.mean()` (never `.sum()`).
4. Linearly interpolate gaps ≤ 5 minutes; leave longer gaps as NaN.

### Stale Detection
Any run of ≥ 15 consecutive minutes where **all 10 channels** change by < 0.01 W simultaneously is tagged as a receiver forward-fill and masked out. A 27-minute real-world flat run (House 1, 2014-08-01) was inspected to confirm this threshold. Flags 1.96% of data across all 20 houses.

### Cycle Detection Thresholds (Frozen in `outputs/selection_frozen.json`)
- ON state: P > 20 W
- Merge gap: ≤ 10 min (mid-cycle pauses)
- Min cycle: ≥ 10 min, ≥ 20 Wh
- Physical ceiling: ≤ 300 min, ≤ 3,500 Wh

### Appliance-Level Flatline Detection
Any single appliance channel constant to < 0.01 W for ≥ 15 consecutive minutes while drawing > 20 W is masked as an isolated transmitter freeze. This catches hardware failures that pass the multivariate stale test (see `DATA.md` Section 2.2 Case Study).

---

## Validation Design

### Why Whole-House Splits?
The target use case is deploying the model on a **new household the model has never seen**. A random row split would give the model access to the test home's cycle signatures during training, making reported metrics unrealistically optimistic. Every metric in this submission is measured on at least one completely unseen home.

### Primary Split
| Split | Houses | Rows |
|:---|:---|:---|
| **Training** | 1, 2, 3, 4 (ch a5), 7–11, 13, 15–21 | 10,841,080 min |
| **Validation** | House 5 | 760,129 min |
| **Test** | House 6 | 688,075 min |

House 4 trains on channel `a5` only (the washing machine with 207 eligible cycles; channel `a4` has only 59 cycles, below the 100-cycle minimum). Houses 12 and 14 are absent from the dataset.

### 4-Fold Leave-One-House-Out Cross-Validation
Each of the four cohort homes (H2, H3, H5, H6) serves as the unseen test house in turn. For each fold, the training set is all remaining eligible houses minus the validation house:

| Fold | Test | Val | Train |
|:---|:---:|:---:|:---|
| H6 | 6 | 5 | All eligible − {5, 6} |
| H2 | 2 | 3 | All eligible − {2, 3} |
| H5 | 5 | 3 | All eligible − {3, 5} |
| H3 | 3 | 6 | All eligible − {3, 6} |

---

## Model Architecture and Trade-Offs

### Why LightGBM?
- Handles missing feature values natively at split time — essential because rolling-window features touching unusable minutes produce NaN rows that should not be silently dropped.
- Gradient-boosted trees naturally capture the non-linear interaction between current aggregate load and its rolling history (e.g. detecting that a 2 kW spike is sustained for > 20 min, not just an isolated kettle burst).
- Fast to train on 10+ million rows without GPU hardware.
- Interpretable: feature importances are available.

### Untested Alternatives
- **XGBoost:** Equivalent architecture but slower on large datasets; NaN routing behaviour differs slightly. Not tested in this submission.
- **Seq2Point CNN (Zhang et al. 2018):** Convolutional sequence-to-point model that treats an input window as a raw 1D signal rather than hand-engineered features. Published results on REFIT show higher F1 on washing machines; not implemented here due to time constraints (see Future Improvements).

### Zero-Clamping
All LightGBM predictions below the validated ON threshold (20 W) are hard-set to 0.0 W. Without this, idle-minute predictions of ~1 W accumulate across 320,000+ off minutes and inflate total-energy error.

### No Appliance Channels as Inputs
The model receives only aggregate household power and timestamp. Dryer, dishwasher, and other sub-metered channels are strictly excluded from all features to simulate real-world deployment where only a whole-house smart meter is available.

---

## Possible Improvements

In priority order, given more time:

1. **Run-length post-processing gate:** Discard any contiguous predicted ON block shorter than 10 minutes or consuming < 25 Wh. A washing machine physically cannot run for 2 minutes. This would eliminate most kettle/microwave false positives without retraining.

2. **Two-stage hurdle architecture:** Decouple detection (binary classifier with balanced classes) from estimation (a regressor trained only on confirmed ON minutes). This removes regression-to-the-mean shrinkage that pulls predicted peaks below the true ~2,000 W heating power.

3. **Wider context window (±60–120 min):** House 6 cycles last a median 129 minutes. The current ±30-minute lag/lead window cannot see the heating pulse that occurred 50 minutes earlier. Adding macro-features (e.g. rolling mean over 120 min, minutes since last 1,500 W aggregate jump) would give the model cycle-length awareness.

4. **Asymmetric or Huber loss:** MSE penalises under-prediction of 2,000 W peaks quadratically: (2000 − 0)² = 4,000,000. This compresses predicted peaks. A quantile or Huber loss would allow predictions to reach true heating-element magnitudes.

5. **Reactive power (power factor):** Washing machine motors are inductive (cos φ ≈ 0.75); electric kettles and ovens are purely resistive (cos φ ≈ 1.0). If the smart meter also recorded VAR, distinguishing a washing machine from a kettle would be far more tractable.

---

## References

All references verified to exist before inclusion. One line states what was adopted or adapted.

| Reference | What was adopted |
|:---|:---|
| Murray, D. et al. (2017). A data management platform for personalised real-time energy feedback. *Scientific Data*, 4, 160122. doi: [10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122) | Primary dataset (REFIT). Sampling interval, channel layout, Issues flag definition, and 8-second nominal polling rate taken directly from this paper. |
| Murray, D. et al. (2015). A data management platform for personalised real-time energy feedback. *EEDAL 2015*. | BST correction methodology and cleaned archive description. |
| Zhang, C. et al. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring. *AAAI 2018*. | The centred-window feature design (lag/lead offsets ±30 min around the target minute) is inspired by the Seq2Point architecture. |
| Kelly, J. & Knottenbelt, W. (2015). Neural NILM: Deep neural networks applied to energy disaggregation. *BuildSys 2015*. | Cross-household evaluation protocol and the finding that unseen-home transfer F1 for washing machines on 1-minute data is typically 0.20–0.35. |
| Klemenjak, C. et al. (2020). Electricity consumption data sets: Pitfalls and opportunities. *arXiv:1912.06200*. | Justification for SAE (sum absolute error of energy) as the primary utility-relevant metric over plain MAE, which is dominated by the 97–98% inactive fraction. |

---

## AI Tools Used

> I used **Claude (Anthropic)** for planning and review, and **Gemini 3.8 Flash in Antigravity** as a coding agent for implementation. I reviewed the code, outputs and documentation, and I can explain the choices made.
