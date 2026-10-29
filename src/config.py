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
    "raw": {
        "file": "REFIT_RAW_081116.7z",
        "urls": [
            "https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z",
            "https://pure.strath.ac.uk/ws/portalfiles/portal/62090177/REFIT_RAW_081116.7z",
        ],
    },
    "clean": {
        "file": "CLEAN_REFIT_081116.7z",
        "urls": [
            "https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z",
            "https://pure.strath.ac.uk/ws/portalfiles/portal/62090184/CLEAN_REFIT_081116.7z",
        ],
    },
}
# The Strathclyde site may block scripted downloads. If so, tell the user to download the two
# archives manually into data/archives/ (Zenodo mirror: DOI 10.5281/zenodo.5063428, record in README).

# (house, appliance_index) of washing-machine channels. Source: MetaData_Tables.xlsx and READMEs.
# House 12 has no washing machine channel.
WM_CANDIDATES = [
    (1, 5), (2, 2), (3, 6), (4, 4), (4, 5), (5, 3), (6, 2), (7, 5), (8, 4), (9, 3),
    (10, 5), (11, 3), (13, 3), (15, 3), (16, 5), (17, 4), (18, 5), (19, 2), (20, 4), (21, 3)
]
TUMBLE_DRYER = {3: 4, 5: 2, 7: 4, 15: 2, 17: 3, 20: 3, 21: 2}
WASHER_DRYER = {1: 4, 8: 3, 9: 2, 18: 4}
DISHWASHER = {
    1: 6, 2: 3, 3: 5, 5: 4, 6: 3, 7: 6, 9: 4, 10: 6, 11: 4, 13: 4, 15: 4, 16: 6, 18: 6, 20: 5, 21: 4
}
H1_CHANNELS = {
    "fridge": 1, "freezer1": 2, "washer_dryer": 4, "washing_machine": 5, "dishwasher": 6
}  # for Section 1 plots
CHANNELS = {2: "a2", 3: "a6", 5: "a3", 6: "a2"}  # cohort washing machine channels
OCCUPANCY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 4, 6: 2, 7: 4, 8: 2, 9: 2, 10: 4,
    11: 1, 12: 3, 13: 4, 15: 1, 16: 6, 17: 3, 18: 2, 19: 4, 20: 2, 21: 4
}
APPLIANCES_OWNED = {
    1: 35, 2: 15, 3: 27, 4: 33, 5: 44, 6: 49, 7: 25, 8: 35, 9: 24, 10: 31,
    11: 25, 12: 26, 13: 28, 15: 19, 16: 48, 17: 22, 18: 34, 19: 26, 20: 39, 21: 23
}
H13_WM_END_UTC = "2015-03-25"  # H13 washing machine changes on this date: use data BEFORE it only
MODEL_EXCLUDE_HOUSES = {4}  # two machines; ambiguous target. Kept in EDA only (as 4a=ch4, 4b=ch5).

# --- PROVISIONAL thresholds (tuned once in Phase 4, then FROZEN) ---
APPLIANCE_SPIKE_W = 4000  # official REFIT rule; used on raw appliance channels
AGG_EXTREME_W = 23000  # ~100 A x 230 V UK supply limit; flag above this (check distribution)
SHORT_GAP_MIN = 5  # interpolate gaps <= this (Section 1)
STALE_MIN = 15  # cleaned data: all channels unchanged for >= this many minutes = stale
ON_W = 20.0  # washing machine ON above this power
MERGE_GAP_MIN = 10  # merge ON segments separated by <= this many minutes (mid-cycle pauses)
MIN_CYCLE_MIN = 10  # discard cycles shorter than this
MIN_CYCLE_WH = 20.0  # discard cycles with less energy than this
MAX_CYCLE_MIN = 300  # discard cycles longer than this (5 hours; physical ceiling)
MAX_CYCLE_WH = 3500.0  # discard cycles with active energy exceeding this (3.5 kWh; physical ceiling)
# eligibility (Phase 4)
MIN_USABLE_DAYS = 90
MIN_CYCLES = 100
SIG_DUR_RANGE_MIN = (20, 240)  # median cycle duration plausible range
SIG_MIN_PEAK_W = 500  # median cycle peak power at least this
