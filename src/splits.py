"""Data splitting logic for NILM disaggregation (Phase 6).

Splits by whole houses to avoid temporal leakage across train/validation/test sets.
Houses are stratified by the presence of high-power thermal confounders (tumble dryers).
"""

from pathlib import Path
import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from . import config as C

# All eligible houses and their WM channels (from outputs/eligibility.csv, Phase 4).
# House 4 uses channel a5 (the 4b washing machine with 207 cycles; a4 has only 59 → excluded).
ALL_ELIGIBLE_CHANNELS: Dict[int, str] = {
    1: "a5",  2: "a2",  3: "a6",  4: "a5",
    5: "a3",  6: "a2",  7: "a5",  8: "a4",
    9: "a3", 10: "a5", 11: "a3", 13: "a3",
   15: "a3", 16: "a5", 17: "a4", 18: "a5",
   19: "a2", 20: "a4", 21: "a3",
}

# Phase 5 EDA cohort (locked)
COHORT_HOUSES = [2, 3, 5, 6]
CHANNELS = {2: "a2", 3: "a6", 5: "a3", 6: "a2"}
CONFOUNDER_HOUSES = {3, 5}     # have tumble dryers
NO_CONFOUNDER_HOUSES = {2, 6}  # no tumble dryers

# Expanded training pool: all eligible houses except test (H6) and validation (H5)
PRIMARY_TEST  = [6]
PRIMARY_VAL   = [5]
PRIMARY_TRAIN = sorted(set(ALL_ELIGIBLE_CHANNELS) - set(PRIMARY_TEST) - set(PRIMARY_VAL))


def get_model_splits() -> Dict[str, Any]:
    """Define the primary train/validation/test split and 4-fold leave-one-house-out splits.

    Training now uses all 17 eligible houses (all eligible - val - test) to maximise cycle
    diversity and allow LightGBM to grow beyond the early-stopping collapse seen when only
    2 training houses were available.
    """
    primary = {
        "train": PRIMARY_TRAIN,   # 17 houses
        "val":   PRIMARY_VAL,     # H5
        "test":  PRIMARY_TEST,    # H6
        "rationale": (
            "Training uses all 17 eligible houses (all eligible minus val H5 and test H6) to "
            "maximise washing-machine cycle diversity across brands, occupancies, and usage habits. "
            "Validation (H5) provides cross-household early stopping and threshold tuning. "
            "Test (H6) evaluates performance on an entirely unseen home with long-duration cycles."
        ),
    }

    # 4-Fold LOHO-CV: each cohort house is the unseen test; train = all eligible minus that
    # test house and one other cohort house used as validation.
    loho_folds = {
        "fold_h6_test": {
            "train": sorted(set(ALL_ELIGIBLE_CHANNELS) - {5, 6}),
            "val": [5], "test": [6],
        },
        "fold_h2_test": {
            "train": sorted(set(ALL_ELIGIBLE_CHANNELS) - {3, 2}),
            "val": [3], "test": [2],
        },
        "fold_h5_test": {
            "train": sorted(set(ALL_ELIGIBLE_CHANNELS) - {3, 5}),
            "val": [3], "test": [5],
        },
        "fold_h3_test": {
            "train": sorted(set(ALL_ELIGIBLE_CHANNELS) - {6, 3}),
            "val": [6], "test": [3],
        },
    }

    # Get date ranges per house from processed parquets (cohort houses only for JSON metadata)
    house_date_ranges = {}
    for h in COHORT_HOUSES:
        df = pd.read_parquet(C.PROC_DIR / f"house_{h}_1min.parquet", columns=["aggregate"])
        house_date_ranges[str(h)] = {
            "channel": CHANNELS[h],
            "first_timestamp_utc": str(df.index[0]),
            "last_timestamp_utc": str(df.index[-1]),
            "total_minutes": len(df),
            "occupancy": C.OCCUPANCY.get(h, "unknown"),
            "appliances_owned": C.APPLIANCES_OWNED.get(h, "unknown"),
            "has_tumble_dryer": h in CONFOUNDER_HOUSES,
        }

    split_config = {
        "cohort_houses": COHORT_HOUSES,
        "cohort_channels": {str(k): v for k, v in CHANNELS.items()},
        "primary_split": primary,
        "loho_folds": loho_folds,
        "house_metadata": house_date_ranges,
    }

    out_path = C.OUT_DIR / "model_split.json"
    with open(out_path, "w") as f:
        json.dump(split_config, f, indent=2)
    print(f"Saved {out_path}")

    return split_config
