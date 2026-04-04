# src/nhisml/subgroup.py
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .preprocess import normalize_weights
from .tasks import make_task


NHIS_MISSING = {7, 8, 9, 97, 98, 99}


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _resolve_run_path(run: str) -> str:
    run = os.path.abspath(run)
    if os.path.isdir(run):
        m = os.path.join(run, "manifest.json")
        if not os.path.exists(m):
            raise FileNotFoundError(f"No manifest.json found in {run}")
        return m
    if run.endswith("manifest.json") and os.path.exists(run):
        return run
    raise FileNotFoundError(f"Run must be a run directory or a manifest.json path: {run}")


def _find_latest_manifest_for_task(runs_dir: str, task: str) -> str:
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = []
    for name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, name)
        manifest_path = os.path.join(run_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            m = _load_json(manifest_path)
            if m.get("task") == task:
                created_at = str(m.get("created_at", ""))
                candidates.append((created_at, manifest_path))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(f"No runs found for task='{task}' under {runs_dir}")

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _resolve_core_path(core_path: Optional[str], year: Optional[int], data_dir: str) -> str:
    if core_path:
        return os.path.abspath(core_path)
    if year is None:
        raise SystemExit("Provide --in CORE_PATH or --year YYYY")
    return os.path.abspath(os.path.join(data_dir, f"core_{int(year)}.parquet"))


def _clean_nhis_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(~s.isin(list(NHIS_MISSING)))


def _ece_weighted(y: np.ndarray, p: np.ndarray, w: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0 or np.sum(w) <= 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    wsum = float(np.sum(w))
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        wm = w[m]
        frac = float(np.sum(wm)) / wsum if wsum > 0 else 0.0
        if frac <= 0:
            continue
        acc = float(np.average(y[m], weights=wm))
        conf = float(np.average(p[m], weights=wm))
        ece += abs(acc - conf) * frac
    return float(ece)


def _weighted_binary_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    out = {
        "weighted_brier": float(brier_score_loss(y, p, sample_weight=w)),
        "weighted_log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "weighted_f1": float(f1_score(y, pred, sample_weight=w)),
        "threshold": float(thr),
        "ece": _ece_weighted(y, p, w, n_bins=10),
    }
    if y.sum() > 0 and (1 - y).sum() > 0:
        out["weighted_auc"] = float(roc_auc_score(y, p, sample_weight=w))
        out["weighted_pr_auc"] = float(average_precision_score(y, p, sample_weight=w))
    else:
        out["weighted_auc"] = float("nan")
        out["weighted_pr_auc"] = float("nan")
    return out


def _recode_sex(df: pd.DataFrame, col: str = "SEX_A") -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    s = _clean_nhis_numeric(df[col])
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[s == 1] = "Male"
    out.loc[s == 2] = "Female"
    return out


def _recode_age_band(df: pd.DataFrame, col: str = "AGEP_A") -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    s = _clean_nhis_numeric(df[col])
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[(s >= 18) & (s <= 34)] = "18-34"
    out.loc[(s >= 35) & (s <= 49)] = "35-49"
    out.loc[(s >= 50) & (s <= 64)] = "50-64"
    out.loc[s >= 65] = "65+"
    return out


def _recode_educ_4(df: pd.DataFrame, col: str = "EDUCP_A") -> pd.Series:
    """
    Collapse EDUCP_A into 4 interpretable levels:
      00-02 -> <HS
      03-04 -> HS/GED
      05-07 -> Some college/AA
      08-10 -> BA+
    """
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    s = _clean_nhis_numeric(df[col])
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[s.isin([0, 1, 2])] = "<HS"
    out.loc[s.isin([3, 4])] = "HS/GED"
    out.loc[s.isin([5, 6, 7])] = "Some college/AA"
    out.loc[s.isin([8, 9, 10])] = "BA+"
    return out


_BUILTINS = {
    "sex": _recode_sex,
    "age": _recode_age_band,
    "education": _recode_educ_4,
}


def _subgroup_table(
    df_eval: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    thr: float,
    subgroup_name: str,
    labels: pd.Series,
    overall: Dict[str, float],
    min_n: int,
    min_pos: int,
    min_neg: int,
) -> pd.DataFrame:
    tmp = pd.DataFrame({"level": labels.astype("object"), "y": y, "p": p, "w": w})
    tmp = tmp[tmp["level"].notna()].copy()

    rows: List[Dict[str, object]] = []
    for lvl, g in tmp.groupby("level", dropna=False):
        yy = g["y"].to_numpy(dtype=int)
        pp = g["p"].to_numpy(dtype=float)
        ww = g["w"].to_numpy(dtype=float)

        n = int(len(g))
        n_pos = int(yy.sum())
        n_neg = int((1 - yy).sum())
        ok = (n >= min_n) and (n_pos >= min_pos) and (n_neg >= min_neg)

        base: Dict[str, object] = {
            "subgroup": subgroup_name,
            "level": str(lvl),
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "weighted_n": float(np.sum(ww)),
            "pos_rate_weighted": float(np.average(yy, weights=ww)) if np.sum(ww) > 0 else float("nan"),
            "threshold": float(thr),
            "meets_min_cell": bool(ok),
        }

        if ok:
            m = _weighted_binary_metrics(yy, pp, ww, thr)
            base.update(m)
            # fairness deltas vs overall
            base["delta_auc"] = float(m["weighted_auc"] - overall["weighted_auc"]) if np.isfinite(overall["weighted_auc"]) else float("nan")
            base["delta_f1"] = float(m["weighted_f1"] - overall["weighted_f1"]) if np.isfinite(overall["weighted_f1"]) else float("nan")
            base["delta_brier"] = float(m["weighted_brier"] - overall["weighted_brier"]) if np.isfinite(overall["weighted_brier"]) else float("nan")
            base["delta_ece"] = float(m["ece"] - overall["ece"]) if np.isfinite(overall["ece"]) else float("nan")
        else:
            base.update({
                "weighted_auc": float("nan"),
                "weighted_pr_auc": float("nan"),
                "weighted_brier": float("nan"),
                "weighted_log_loss": float("nan"),
                "weighted_f1": float("nan"),
                "ece": float("nan"),
                "delta_auc": float("nan"),
                "delta_f1": float("nan"),
                "delta_brier": float("nan"),
                "delta_ece": float("nan"),
            })

        rows.append(base)

    out = pd.DataFrame(rows).sort_values(["subgroup", "n"], ascending=[True, False]).reset_index(drop=True)
    return out


def cli(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser("nhisml subgroup")

    p.add_argument("--run", default=None, help="Run directory or manifest.json path")
    p.add_argument("--task", default=None, help="Task name (used with --latest)")
    p.add_argument("--latest", action="store_true", help="Use latest run for the given --task")
    p.add_argument("--runs-dir", default="runs", help="Base runs directory (default: runs/)")

    p.add_argument("--in", dest="core_path", default=None, help="Core parquet to evaluate on (optional if using --year)")
    p.add_argument("--year", type=int, default=None, help="Shortcut for --in <data-dir>/core_YYYY.parquet")
    p.add_argument("--data-dir", default="data", help="Base data directory for --year shortcut (default: data/)")

    p.add_argument("--out", default=None, help="Optional output csv path (default: <run_dir>/subgroups_task=<task>.csv)")
    p.add_argument("--by", nargs="+", required=True, help="Subgroups: sex age education or raw columns like REGION URBRRL23 EDUCP_A")
    p.add_argument("--weight-col", default="WTFA_A")
    p.add_argument("--threshold-key", default=None, help="Threshold key (default: manifest['model'])")
    p.add_argument("--min-n", type=int, default=200)
    p.add_argument("--min-pos", type=int, default=25)
    p.add_argument("--min-neg", type=int, default=25)
    args = p.parse_args(argv)

    # Resolve manifest
    if args.latest:
        if not args.task:
            raise SystemExit("--latest requires --task")
        if args.run:
            raise SystemExit("Use either --run OR (--task with --latest), not both")
        manifest_path = _find_latest_manifest_for_task(args.runs_dir, args.task)
    else:
        if not args.run:
            raise SystemExit("Provide --run or use --task with --latest")
        manifest_path = _resolve_run_path(args.run)

    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    manifest = _load_json(manifest_path)
    task = make_task(manifest["task"])

    model_path = os.path.join(run_dir, manifest["artifacts"]["model"])
    model = joblib.load(model_path)

    thresholds_path = os.path.join(run_dir, manifest["artifacts"]["thresholds"])
    thresholds = _load_json(thresholds_path)

    model_name = manifest["model"]
    thr_key = args.threshold_key or model_name
    thr = float(thresholds.get(thr_key, 0.5))

    core_path = _resolve_core_path(args.core_path, args.year, args.data_dir)
    df = pd.read_parquet(core_path)

    # labels + eligibility
    y, eligible = task.make_labels(df)
    if eligible is None:
        eligible = np.ones(len(df), dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)

    df_eval = df.loc[eligible].reset_index(drop=True)
    y_eval = np.asarray(y)[eligible].astype(int)

    # weights
    if args.weight_col in df_eval.columns:
        w_eval = normalize_weights(df_eval[args.weight_col])
        used_weights = True
    else:
        w_eval = np.ones(len(df_eval), dtype=float)
        used_weights = False

    # predict
    p1 = model.predict_proba(df_eval)[:, 1]

    # overall metrics baseline for deltas
    overall = _weighted_binary_metrics(y_eval, p1, w_eval, thr)

    tables: List[pd.DataFrame] = []
    skipped: List[str] = []

    for item in args.by:
        key = item.strip()
        key_l = key.lower()

        if key_l in _BUILTINS:
            labels = _BUILTINS[key_l](df_eval)
            if labels.isna().all():
                skipped.append(key)
                continue
            tables.append(_subgroup_table(
                df_eval=df_eval,
                y=y_eval,
                p=p1,
                w=w_eval,
                thr=thr,
                subgroup_name=key_l,
                labels=labels,
                overall=overall,
                min_n=args.min_n,
                min_pos=args.min_pos,
                min_neg=args.min_neg,
            ))
            continue

        # raw column
        if key not in df_eval.columns:
            skipped.append(key)
            continue
        labels = df_eval[key].astype("object")
        tables.append(_subgroup_table(
            df_eval=df_eval,
            y=y_eval,
            p=p1,
            w=w_eval,
            thr=thr,
            subgroup_name=key,
            labels=labels,
            overall=overall,
            min_n=args.min_n,
            min_pos=args.min_pos,
            min_neg=args.min_neg,
        ))

    if not tables:
        raise SystemExit(f"No valid subgroup specs. Missing/empty: {skipped}")

    out = pd.concat(tables, axis=0, ignore_index=True)

    out_path = args.out or os.path.join(run_dir, f"subgroups_task={task.name}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Subgroup eval: {task.name} | Model: {model_name} | Threshold: {thr:.3f} | Weights: {'yes' if used_weights else 'no'}")
    if skipped:
        print(f"[subgroup] Skipped (missing): {', '.join(skipped)}")
    print(f"Wrote: {os.path.relpath(out_path, start=run_dir)}")
    print(out.head(12).to_string(index=False))
