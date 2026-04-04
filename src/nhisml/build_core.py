from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from .featuresets import get_featureset
from .tasks import make_task


NHIS_MISSING_CODES = {7, 8, 9, 97, 98, 99}


def _raw_zip_path(data_dir: str, year: int) -> Path:
    yy = str(year)[-2:]
    return Path(data_dir) / "raw" / str(year) / f"adult{yy}csv.zip"


def _read_adult_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing raw zip: {zip_path}. Run `nhisml fetch --year {zip_path.parent.name}` first.")

    with zipfile.ZipFile(zip_path, "r") as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError(f"No CSV found in {zip_path}")
        # prefer adult*.csv if present
        csvs.sort()
        candidates = [c for c in csvs if "adult" in c.lower()]
        fname = candidates[0] if candidates else csvs[0]
        with zf.open(fname) as f:
            return pd.read_csv(f, low_memory=False)


def _collect_required_columns(
    featureset: str,
    tasks: List[str],
    weight_col: str,
    extra_cols: Optional[List[str]] = None,
) -> List[str]:
    fs = get_featureset(featureset)

    required: Set[str] = set(fs.all_columns)
    required.add(weight_col)

    required.update({"SEX_A", "AGEP_A", "EDUCP_A"})

    for t in tasks:
        task = make_task(t)
        required.update(task.required_cols)

    if extra_cols:
        required.update(extra_cols)

    return sorted(required)


def _basic_normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()

    # Targets like PHSTAT_A, SMKCIGST_A, SMKNOW_to numeric.
    for c in ["PHSTAT_A", "SMKCIGST_A", "SMKNOW_A", "WTFA_A"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def build_core_year(
    year: int,
    data_dir: str = "data",
    out_dir: str = "data",
    featureset: str = "core",
    tasks: Optional[List[str]] = None,
    weight_col: str = "WTFA_A",
    extra_cols: Optional[List[str]] = None,
) -> Path:
    tasks = tasks or ["srh_binary", "smoking_current"]

    zip_path = _raw_zip_path(data_dir, year)
    df = _read_adult_csv_from_zip(zip_path)

    cols = _collect_required_columns(featureset, tasks, weight_col, extra_cols=extra_cols)
    core = _basic_normalize(df, cols)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_p / f"core_{year}.parquet"
    core.to_parquet(out_path, index=False)

    manifest = {
        "year": year,
        "featureset": featureset,
        "tasks": tasks,
        "weight_col": weight_col,
        "n_rows": int(len(core)),
        "n_cols": int(core.shape[1]),
        "columns": list(core.columns),
        "source_zip": str(zip_path),
    }
    with open(out_dir_p / f"core_{year}.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[build-core] Wrote {out_path} ({core.shape[0]} rows, {core.shape[1]} cols)")
    return out_path


def cli(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser("nhisml build-core")
    p.add_argument("--year", type=int, action="append", required=True)
    p.add_argument("--data-dir", default="data", help="Base data directory (raw zips under data/raw/)")
    p.add_argument("--out-dir", default="data", help="Output directory for core parquet")
    p.add_argument("--featureset", default="core")
    p.add_argument("--task", action="append", default=None, help="Task(s) to ensure label cols are included")
    p.add_argument("--weight-col", default="WTFA_A")
    args = p.parse_args(argv)

    for y in args.year:
        build_core_year(
            year=y,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            featureset=args.featureset,
            tasks=args.task or ["srh_binary", "smoking_current"],
            weight_col=args.weight_col,
        )
