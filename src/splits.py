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

# Target cohort locked in Phase 5
COHORT_HOUSES = [2, 3, 5, 6]
CHANNELS = {2: "a2", 3: "a6", 5: "a3", 6: "a2"}
CONFOUNDER_HOUSES = {3, 5}  # have tumble dryers
NO_CONFOUNDER_HOUSES = {2, 6}  # no tumble dryers


def get_model_splits() -> Dict[str, Any]:
    """Define the primary train/validation/test split and 4-fold leave-one-house-out splits."""
    # Primary split:
    # Train: House 2 (no confounder, 4 occ) and House 3 (confounder, 2 occ)
    # Validation: House 5 (confounder, 4 occ)
    # Test: House 6 (no confounder, 2 occ)
    primary = {
        "train": [2, 3],
        "val": [5],
        "test": [6],
        "rationale": (
            "Train houses (H2, H3) provide one home with a tumble dryer and one without, "
            "spanning both 2-person and 4-person occupancies. Validation (H5) tests generalization "
            "to a high-frequency, complex home with a tumble dryer for early stopping and threshold tuning. "
            "Test (H6) evaluates performance on an entirely unseen home with prolonged eco-cycles."
        ),
    }

    # 4-Fold Leave-One-House-Out Cross-Validation (each house as unseen test)
    loho_folds = {
        "fold_h6_test": {"train": [2, 3], "val": [5], "test": [6]},
        "fold_h2_test": {"train": [5, 6], "val": [3], "test": [2]},
        "fold_h5_test": {"train": [2, 6], "val": [3], "test": [5]},
        "fold_h3_test": {"train": [2, 5], "val": [6], "test": [3]},
    }

    # Get date ranges per house from processed parquets
    house_date_ranges = {}
    for h in COHORT_HOUSES:
        df = pd.read_parquet(C.PROC_DIR / f"house_{h}_1min.parquet", columns=["aggregate"])
        house_date_ranges[str(h)] = {
            "channel": CHANNELS[h],
            "first_timestamp_utc": str(df.index[0]),
            "last_timestamp_utc": str(df.index[-1]),
            "total_minutes": len(df),
            "occupancy": C.OCCUPANCY[h],
            "appliances_owned": C.APPLIANCES_OWNED[h],
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
