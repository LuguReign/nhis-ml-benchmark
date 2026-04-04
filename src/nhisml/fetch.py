from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from tqdm import tqdm


NHIS_URLS: Dict[int, str] = {
    2023: "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip",
    2024: "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/adult24csv.zip",
}


def _default_zip_path(data_dir: str, year: int) -> Path:
    yy = str(year)[-2:]
    return Path(data_dir) / "raw" / str(year) / f"adult{yy}csv.zip"


def download_file(url: str, out_path: Path, force: bool = False, timeout: int = 120) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[fetch] Using cached file: {out_path}")
        return out_path

    print(f"[fetch] Downloading: {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

        with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        tmp_path.replace(out_path)

    print(f"[fetch] Saved: {out_path}")
    return out_path


def fetch_year(year: int, data_dir: str = "data", force: bool = False, url: Optional[str] = None) -> Path:
    if url is None:
        url = NHIS_URLS.get(year)
    if not url:
        raise ValueError(f"No URL configured for year {year}. Provide --url to override.")

    out_path = _default_zip_path(data_dir, year)
    return download_file(url, out_path, force=force)


def cli(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser("nhisml fetch")
    p.add_argument("--year", type=int, action="append", required=True, help="Year(s) to fetch, e.g. --year 2023 --year 2024")
    p.add_argument("--data-dir", default="data", help="Base data directory (default: data/)")
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    p.add_argument("--url", default=None, help="Override download URL (applies only if one year is provided)")
    args = p.parse_args(argv)

    if args.url and len(args.year) != 1:
        raise ValueError("--url override is only supported when a single --year is provided.")

    for y in args.year:
        fetch_year(y, data_dir=args.data_dir, force=args.force, url=args.url)
