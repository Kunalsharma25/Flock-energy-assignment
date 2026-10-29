"""Prepare 1-minute Parquet files from official cleaned REFIT CSV data.

Extracts one house at a time to data/tmp/, resamples to 1-minute bins (mean for power,
max for issues), saves to data/processed/house_{N}_1min.parquet, and deletes temp CSV.
"""

from pathlib import Path
import sys
import argparse
import time
import py7zr
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config as C
from src.cleaning import resample_to_1min

# All available cleaned houses in REFIT (1-21, no House 14) in archive order for fast sequential decompression
ARCHIVE_HOUSE_ORDER = [1, 10, 11, 12, 13, 15, 16, 17, 18, 19, 2, 20, 21, 3, 4, 5, 6, 7, 8, 9]
ALL_HOUSES = sorted(ARCHIVE_HOUSE_ORDER)


def process_house(house_id: int, archive_path: Path, tmp_dir: Path, proc_dir: Path, force: bool = False) -> Path:
    """Extract, resample to 1-min, save parquet, and clean up temp CSV for a single house."""
    out_parquet = proc_dir / f"house_{house_id}_1min.parquet"
    if out_parquet.exists() and not force:
        print(f"[SKIP] House {house_id}: {out_parquet.name} already exists ({out_parquet.stat().st_size / (1024*1024):.2f} MB)")
        return out_parquet

    target_csv_name = f"CLEAN_House{house_id}.csv"
    temp_csv_path = tmp_dir / target_csv_name

    t0 = time.time()
    # 1. Extract only this house CSV if not already in tmp
    if not temp_csv_path.exists():
        print(f"[EXTRACT] House {house_id}: Extracting {target_csv_name} from {archive_path.name}...")
        t_ext = time.time()
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extract(path=tmp_dir, targets=[target_csv_name])
        print(f"  Extracted {temp_csv_path.stat().st_size / (1024*1024):.1f} MB in {time.time() - t_ext:.1f}s")
    else:
        print(f"[FOUND] House {house_id}: Using existing {temp_csv_path.name}")

    # 2. Read with usecols by header name, downcast types
    print(f"[PROCESS] House {house_id}: Reading CSV and resampling to 1-minute...")
    t_proc = time.time()
    usecols = ["Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)] + ["Issues"]
    dtypes = {
        "Aggregate": "float32",
        **{f"Appliance{i}": "float32" for i in range(1, 10)},
        "Issues": "int8",
    }
    df = pd.read_csv(temp_csv_path, usecols=usecols, dtype=dtypes)
    raw_rows = len(df)

    # 3. Drop missing Unix, sort, set UTC DatetimeIndex
    df = df.dropna(subset=["Unix"])
    df = df.sort_values("Unix")
    df.index = pd.to_datetime(df["Unix"], unit="s", utc=True)
    df.index.name = "timestamp"
    df = df.drop(columns=["Unix"])

    # Rename columns to standard lowercase
    rename_map = {
        "Aggregate": "aggregate",
        **{f"Appliance{i}": f"a{i}" for i in range(1, 10)},
        "Issues": "issues",
    }
    df = df.rename(columns=rename_map)

    # 4. Resample to 1 min (mean for power, max for issues)
    resampled = resample_to_1min(df, issues_col="issues")

    # 5. Save to parquet
    proc_dir.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(out_parquet, engine="pyarrow", compression="snappy")

    # 6. Delete temp CSV
    if temp_csv_path.exists():
        temp_csv_path.unlink()

    elapsed = time.time() - t0
    first_ts = resampled.index[0]
    last_ts = resampled.index[-1]
    print(f"[DONE] House {house_id}: {raw_rows:,} raw rows -> {len(resampled):,} 1-min rows")
    print(f"  Coverage: {first_ts} to {last_ts} | Parquet: {out_parquet.stat().st_size / (1024*1024):.2f} MB | Time: {elapsed:.1f}s")

    return out_parquet


def main():
    parser = argparse.ArgumentParser(description="Prepare 1-minute Parquet files for REFIT cleaned houses.")
    parser.add_argument(
        "--houses",
        type=int,
        nargs="+",
        default=ALL_HOUSES,
        help="List of house numbers to process (default: all 20 houses)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-extraction and processing")
    args = parser.parse_args()

    clean_archive = C.ARCHIVE_DIR / C.ARCHIVES["clean"]["file"]
    if not clean_archive.exists():
        print(f"[ERROR] Clean archive not found at {clean_archive}. Run scripts/download_data.py first.")
        sys.exit(1)

    C.TMP_DIR.mkdir(parents=True, exist_ok=True)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Sort houses by archive order for optimal extraction performance
    houses_to_run = [h for h in ARCHIVE_HOUSE_ORDER if h in args.houses]
    print(f"Starting data preparation for {len(houses_to_run)} houses in archive order: {houses_to_run}")
    start_total = time.time()

    for h in houses_to_run:
        process_house(h, clean_archive, C.TMP_DIR, C.PROC_DIR, force=args.force)

    # Ensure tmp directory is clean
    remaining_tmp = list(C.TMP_DIR.glob("*.csv"))
    for f in remaining_tmp:
        f.unlink()

    print("\n" + "=" * 60)
    print(f"ALL HOUSES PROCESSED in {time.time() - start_total:.1f}s")
    print(f"Processed parquets directory: {C.PROC_DIR.resolve()}")
    print(f"data/tmp is empty: {len(list(C.TMP_DIR.iterdir())) == 0}")
    print("=" * 60)


if __name__ == "__main__":
    main()
