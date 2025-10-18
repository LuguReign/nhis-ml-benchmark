#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHIS downloader + Parquet builder (Adults 2023/2024)

Library usage:
    from nhis_data import fetch, fetch2324, fetch24
    p24 = fetch(2024)
    p23, p24 = fetch2324()

CLI:
    python src/nhis_data.py           # fetches 2023 & 2024
    python src/nhis_data.py 2024      # just 2024
    python src/nhis_data.py 2023 2024 # explicit years
"""

from __future__ import annotations
import os, re, zipfile, argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

NHIS_URLS = {
    2024: "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/adult24csv.zip",
    2023: "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip",
}

# Feature list
FEATURES = [
    # SES
    "RATCAT_A","POVRATTC_A",
    # Education
    "EDUCP_A","MAXEDUCP_A",
    # Employment
    "EMPWKHRS3_A","EMPWRKFT1_A","EMPHEALINS_A","EMPSICKLV_A","EMPLASTWK_A",
    "EMPNOWRK_A","EMPWHENWRK_A","EMDSUPER_A",
    # Social
    "MARITAL_A","MARSTAT_A","LONELY_A","SUPPORT_A","URBRRL23","REGION",
    "FDSCAT3_A","FDSCAT4_A",
    # Disability & Functioning
    "DISAB3_A","ANYDIFF_A","DIFF_A","COGMEMDFF_A","COMDIFF_A","VISIONDF_A","HEARINGDF_A",
    # Mental health
    "K6SPD_A","WORTHLESS_A","HOPELESS_A","SAD_A","NERVOUS_A","RESTLESS_A","EFFORT_A",
    "DEPFREQ_A","ANXFREQ_A","DEPLEVEL_A","DEPMED_A","ANXMED_A","MHRX_A","MHTHRPY_A",
    "MHTHDLY_A","MHTHND_A",
    # Conditions
    "HYPEV_A","DIBEV_A","CHDEV_A","MIEV_A","STREV_A","ANGEV_A",
    "ASEV_A","ASTILL_A","ARTHEV_A","COPDEV_A","CANEV_A",
    # Chronic
    "CHLEV_A","CHL12M_A","HYP12M_A","HYPMED_A","KIDWEAKEV_A","LIVEREV_A","HEPEV_A",
    "CROHNSEV_A","ULCCOLEV_A","PSOREV_A","CFSNOW_A",
    # Access
    "HICOV_A","USUALPL_A","MEDNG12M_A","MEDDL12M_A","RXDG12M_A","LASTDR_A","WELLVIS_A",
]
TARGET_RAW = "PHSTAT_A"
WEIGHT_COL = "WTFA_A"

def _download_with_progress(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req) as r:
        total = r.length or 0
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
        with open(dest, "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                if bar: bar.update(len(b))
        if bar: bar.close()
    return dest

def _safe_unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir

def _find_adult_csv(root: Path, year: int) -> Optional[Path]:
    patt = re.compile(rf"adult{str(year)[-2:]}\.csv$", flags=re.I)
    for p in root.rglob("*.csv"):
        if patt.search(p.name):
            return p
    for p in root.rglob("adult.csv"):
        return p
    return None

def _year_to_out_parquet(year: int) -> Path:
    return Path("data") / f"Adults{str(year)[-2:]}_corefeatures.parquet"

def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({7: np.nan, 8: np.nan, 9: np.nan, 97: np.nan, 98: np.nan, 99: np.nan})
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    return df

def build_core_parquet_from_csv(csv_path: Path, year: int) -> Path:
    print(f"[build] Reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    keep_cols = list({*FEATURES, TARGET_RAW, WEIGHT_COL} & set(df.columns))
    missing = sorted(set([*FEATURES, TARGET_RAW, WEIGHT_COL]) - set(keep_cols))
    if missing:
        print(f"[warn] {len(missing)} expected cols missing (ok if not in this year): {missing[:10]}{'...' if len(missing)>10 else ''}")
    out_df = _basic_clean(df[keep_cols].copy())
    out_path = _year_to_out_parquet(year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[ok] Saved {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} cols)")
    return out_path

def ensure_year_and_build_parquet(year: int, cache_dir: Path = Path("data") / "raw") -> Tuple[Path, Path]:
    if year not in NHIS_URLS:
        raise ValueError(f"Unsupported year {year}. Supported: {sorted(NHIS_URLS)}")
    url = NHIS_URLS[year]
    year_dir = cache_dir / str(year)
    zip_path = year_dir / Path(url).name

    if not zip_path.exists():
        print(f"[dl] {year}: downloading {url}")
        year_dir.mkdir(parents=True, exist_ok=True)
        _download_with_progress(url, zip_path)
    else:
        print(f"[dl] {year}: zip exists at {zip_path}")

    extract_root = year_dir / "unzipped"
    _safe_unzip(zip_path, extract_root)

    csv_path = _find_adult_csv(extract_root, year)
    if csv_path is None:
        raise FileNotFoundError(f"adult CSV not found under {extract_root}")

    parquet_path = build_core_parquet_from_csv(csv_path, year)
    return csv_path, parquet_path

# -------- Short, friendly aliases --------
def fetch(year: int) -> Path:
    """Download + build Parquet for a year. Returns Parquet path."""
    _, parquet = ensure_year_and_build_parquet(year)
    return parquet

def fetch23() -> Path:
    return fetch(2023)

def fetch24() -> Path:
    return fetch(2024)

def fetch2324() -> list[Path]:
    return [fetch(2023), fetch(2024)]

def fetch_all() -> list[Path]:
    return fetch2324()

__all__ = [
    "fetch", "fetch23", "fetch24", "fetch2324", "fetch_all",
    "ensure_year_and_build_parquet", "build_core_parquet_from_csv",
    "FEATURES", "TARGET_RAW", "WEIGHT_COL"
]

# ------------- CLI -------------
def _cli():
    ap = argparse.ArgumentParser(description="Fetch NHIS Adults (2023/2024) and build Parquet.")
    ap.add_argument("years", nargs="*", type=int, help="Years to fetch, e.g., 2023 2024. Default: both.")
    ap.add_argument("--cache_dir", type=str, default="data/raw", help="Where to cache zips/extracts.")
    args = ap.parse_args()

    years = args.years if args.years else [2023, 2024]
    for y in years:
        try:
            csv_path, pq_path = ensure_year_and_build_parquet(y, Path(args.cache_dir))
            print(f"[done] {y}: CSV={csv_path} PARQUET={pq_path}")
        except Exception as e:
            print(f"[error] {y}: {e}")
            raise

if __name__ == "__main__":
    _cli()
