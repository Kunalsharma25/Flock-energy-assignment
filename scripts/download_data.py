"""Download REFIT archives if not already present.

Skips downloads if archives exist in data/archives/.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
from src import config as C


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download a file streaming to a .part file, then rename."""
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        print(f"Connecting to {url}...")
        with requests.get(url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(part_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\rDownloaded {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="")
                        else:
                            print(f"\rDownloaded {downloaded / (1024*1024):.1f} MB", end="")
            print()
        if part_path.exists():
            part_path.rename(dest_path)
            print(f"Successfully downloaded {dest_path.name}")
            return True
    except Exception as e:
        print(f"\nDownload failed from {url}: {e}")
        if part_path.exists():
            part_path.unlink()
        return False
    return False


def main():
    C.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    all_success = True

    for key, info in C.ARCHIVES.items():
        archive_name = info["file"]
        dest_path = C.ARCHIVE_DIR / archive_name

        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"[SKIP] Archive '{archive_name}' already exists ({dest_path.stat().st_size / (1024*1024):.1f} MB).")
            continue

        print(f"\nArchive '{archive_name}' not found. Attempting download...")
        downloaded = False
        for url in info["urls"]:
            if download_file(url, dest_path):
                downloaded = True
                break

        if not downloaded:
            print(f"\n[ERROR] Failed to download '{archive_name}' from all available URLs.")
            print("Please download manually and place the archive in:")
            print(f"  {C.ARCHIVE_DIR.resolve()}")
            print("Zenodo mirror DOI: 10.5281/zenodo.5063428")
            all_success = False

    if not all_success:
        sys.exit(1)
    print("\nAll archives are present.")


if __name__ == "__main__":
    main()
