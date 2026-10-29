"""Inspect raw and clean REFIT archives, dump listings, and check assumptions."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config as C
from src.data_io import dump_archive_listing, extract_specific_files, read_first_n_lines_from_archive


def main():
    raw_archive = C.ARCHIVE_DIR / C.ARCHIVES["raw"]["file"]
    clean_archive = C.ARCHIVE_DIR / C.ARCHIVES["clean"]["file"]

    print("Inspecting archives and dumping listings...")
    raw_listing_file = C.OUT_DIR / "archive_listing_raw.txt"
    clean_listing_file = C.OUT_DIR / "archive_listing_clean.txt"

    raw_info = dump_archive_listing(raw_archive, raw_listing_file)
    print(f"Saved raw archive listing to {raw_listing_file}")

    clean_info = dump_archive_listing(clean_archive, clean_listing_file)
    print(f"Saved clean archive listing to {clean_listing_file}")

    # Extract House 1 raw files if not already present
    h1_raw_files = [f["filename"] for f in raw_info["files"] if "House1_" in f["filename"] or "House_1_" in f["filename"]]
    print(f"\nHouse 1 raw files detected: {h1_raw_files}")
    missing_raw = [f for f in h1_raw_files if not (C.RAW_DIR / f).exists()]
    if missing_raw:
        print(f"Extracting {missing_raw} to data/raw/ ...")
        extract_specific_files(raw_archive, missing_raw, C.RAW_DIR)
        print("Raw House 1 extraction complete.")
    else:
        print("Raw House 1 files already extracted in data/raw/.")

    # Read first 3 lines of raw House 1 Part 1
    raw_h1_part1_path = C.RAW_DIR / h1_raw_files[0]
    with open(raw_h1_part1_path, "r", encoding="utf-8") as f:
        raw_first_lines = [f.readline().strip() for _ in range(3)]

    # Read first 3 lines of cleaned House 1 without extracting full archive
    print("\nReading first 3 lines of CLEAN_House1.csv from clean archive...")
    clean_first_lines = read_first_n_lines_from_archive(clean_archive, "CLEAN_House1.csv", n_lines=3)

    print("\n" + "=" * 60)
    print("ARCHIVE INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Raw Archive:   {raw_info['size_bytes'] / (1024*1024):.2f} MB, Solid={raw_info['solid']}, Files={raw_info['num_files']}, Uncompressed={raw_info['total_uncompressed'] / (1024**3):.2f} GB")
    print(f"Clean Archive: {clean_info['size_bytes'] / (1024*1024):.2f} MB, Solid={clean_info['solid']}, Files={clean_info['num_files']}, Uncompressed={clean_info['total_uncompressed'] / (1024**3):.2f} GB")
    print(f"\nRaw House 1 Files: {h1_raw_files}")
    print("\nFirst 3 lines of RAW_House1_Part1.csv:")
    for line in raw_first_lines:
        print("  ", line)
    print("\nFirst 3 lines of CLEAN_House1.csv:")
    for line in clean_first_lines:
        print("  ", line)
    print("=" * 60)


if __name__ == "__main__":
    main()
