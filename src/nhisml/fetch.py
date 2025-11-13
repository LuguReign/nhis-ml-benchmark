# src/nhisml/fetch.py
from __future__ import annotations
import argparse, os, io, zipfile, tempfile
import requests
import pandas as pd
from typing import Dict, List
from .featuresets import get_featureset

NHIS_URLS: Dict[int, str] = {
    2023: "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip",
    2024: "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/adult24csv.zip",
}

def _download_zip(url: str) -> bytes:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        return r.content

def _extract_single_csv(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        # pick the first CSV that looks like adult*.csv
        candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError("No CSV found in NHIS zip.")
        # prefer top-level adult*.csv if present
        candidates.sort()
        target = [c for c in candidates if "adult" in c.lower()]
        fname = (target[0] if target else candidates[0])
        with zf.open(fname) as f:
            df = pd.read_csv(f, low_memory=False)
    return df

def fetch_year(year: int, outdir: str = "data/processed", featureset: str = "srh_core") -> str:
    url = NHIS_URLS.get(year)
    if not url:
        raise ValueError(f"No URL configured for year {year}")
    os.makedirs(outdir, exist_ok=True)
    print(f"[fetch] Downloading NHIS Adults {year} ...")
    zbytes = _download_zip(url)
    print(f"[fetch] Extracting CSV ...")
    df = _extract_single_csv(zbytes)
    print(f"[fetch] CSV shape: {tuple(df.shape)}")

    feats = get_featureset(featureset)
    keep = [c for c in feats.all_columns if c in df.columns]
    core = df[keep].copy()
    out = os.path.join(outdir, f"Adults{str(year)[-2:]}_core.parquet")
    core.to_parquet(out, index=False)
    print(f"[fetch] Saved {out} ({core.shape[0]} rows, {core.shape[1]} cols)")
    return out

def cli():
    p = argparse.ArgumentParser("nhis-fetch")
    p.add_argument("years", nargs="+", type=int, help="2023 2024")
    p.add_argument("--outdir", default="data/processed")
    args = p.parse_args()
    for yr in args.years:
        fetch_year(yr, outdir=args.outdir)
