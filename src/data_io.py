"""Data I/O utilities for REFIT NILM project.

Handles archive listing, streaming extraction using py7zr, and parquet I/O.
"""

from pathlib import Path
from typing import Dict, List, Any
import py7zr
import pandas as pd

from . import config as C


def get_archive_info(archive_path: Path) -> Dict[str, Any]:
    """Get metadata and file listing for a .7z archive using py7zr."""
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        info = z.archiveinfo()
        files = []
        for f in z.list():
            files.append({
                "filename": f.filename,
                "uncompressed": f.uncompressed,
                "compressed": getattr(f, "compressed", None),
                "is_directory": f.is_directory,
                "creationtime": str(getattr(f, "creationtime", "")),
            })
        return {
            "archive_path": str(archive_path),
            "size_bytes": info.stat.st_size,
            "solid": info.solid,
            "blocks": info.blocks,
            "methods": info.method_names,
            "total_uncompressed": info.uncompressed,
            "num_files": len(files),
            "files": files,
        }


def dump_archive_listing(archive_path: Path, output_file: Path) -> Dict[str, Any]:
    """Inspect archive and write detailed listing to output_file."""
    info = get_archive_info(archive_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"Archive: {Path(info['archive_path']).name}\n")
        out.write(f"Archive Size: {info['size_bytes']:,} bytes ({info['size_bytes'] / (1024*1024):.2f} MB)\n")
        out.write(f"Solid Archive: {info['solid']}\n")
        out.write(f"Blocks: {info['blocks']}\n")
        out.write(f"Compression Methods: {', '.join(info['methods'])}\n")
        out.write(f"Total Files: {info['num_files']}\n")
        out.write(f"Total Uncompressed Size: {info['total_uncompressed']:,} bytes ({info['total_uncompressed'] / (1024*1024*1024):.2f} GB)\n")
        out.write("-" * 80 + "\n")
        out.write(f"{'Filename':<35} {'Uncompressed (bytes)':<22} {'Size (MB)':<12}\n")
        out.write("-" * 80 + "\n")
        for f in sorted(info["files"], key=lambda x: x["filename"]):
            mb = f["uncompressed"] / (1024 * 1024)
            out.write(f"{f['filename']:<35} {f['uncompressed']:<22,} {mb:<12.2f}\n")
    return info


def extract_specific_files(archive_path: Path, targets: List[str], dest_dir: Path) -> List[Path]:
    """Extract only specified targets from a .7z archive into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extract(path=dest_dir, targets=targets)
    extracted = [dest_dir / t for t in targets]
    return extracted


def read_first_n_lines_from_archive(archive_path: Path, filename: str, n_lines: int = 5, tmp_dir: Path = None) -> List[str]:
    """Read first n lines of a file by extracting to tmp_dir and deleting after."""
    if tmp_dir is None:
        tmp_dir = C.TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target_path = tmp_dir / filename
    if not target_path.exists():
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extract(path=tmp_dir, targets=[filename])
    lines = []
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(n_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\r\n"))
    return lines
