from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import fetch as fetch_mod
from . import build_core as build_core_mod
from . import train as train_mod
from . import evaluate as evaluate_mod
from . import subgroup as subgroup_mod
from .featuresets import list_featuresets, get_featureset
from .tasks import list_tasks, make_task


def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(prog="nhisml", description="nhisml: survey-aware NHIS Adults ML starter kit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # fetch
    pf = sub.add_parser("fetch", help="Download and cache NHIS Adults raw zip files")
    pf.add_argument("--year", type=int, action="append", required=True, help="Year(s) to fetch, e.g. --year 2023 --year 2024")
    pf.add_argument("--data-dir", default="data", help="Base data directory (default: data/)")
    pf.add_argument("--force", action="store_true", help="Re-download even if cached")
    pf.add_argument("--url", default=None, help="Override download URL (only if one --year is provided)")
    pf.set_defaults(func=lambda a: fetch_mod.cli(_rebuild_argv("fetch", a)))

    # build-core
    pc = sub.add_parser("build-core", help="Build a clean core parquet from cached raw NHIS zip")
    pc.add_argument("--year", type=int, action="append", required=True)
    pc.add_argument("--data-dir", default="data", help="Base data directory (raw zips under data/raw/)")
    pc.add_argument("--out-dir", default="data", help="Output directory for core parquet")
    pc.add_argument("--featureset", default="core")
    pc.add_argument("--task", action="append", default=None, help="Task(s) to ensure label cols are included")
    pc.add_argument("--weight-col", default="WTFA_A")
    pc.set_defaults(func=lambda a: build_core_mod.cli(_rebuild_argv("build-core", a)))

    # train
    pt = sub.add_parser("train", help="Train a baseline model for a task and produce a run directory")
    pt.add_argument("--in", dest="core_path", required=True, help="Path to core parquet (e.g., data/core_2023.parquet)")
    pt.add_argument("--task", default="srh_binary")
    pt.add_argument("--featureset", default="core")
    pt.add_argument("--model", default="lasso", help="lasso | rf")
    pt.add_argument("--run-dir", default="runs")
    pt.add_argument("--calibrate", action="store_true")
    pt.add_argument("--weight-col", default="WTFA_A")
    pt.set_defaults(func=lambda a: train_mod.cli(_rebuild_argv("train", a)))

    # evaluate
    pe = sub.add_parser("evaluate", help="Evaluate a trained run on a core parquet (writes metrics + predictions)")
    pe.add_argument("--run", required=False, help="Run directory or manifest.json path")
    pe.add_argument("--task", default=None, help="Task name (used with --latest)")
    pe.add_argument("--latest", action="store_true", help="Use latest run for the given --task")
    pe.add_argument("--runs-dir", default="runs", help="Base runs directory (default: runs/)")

    pe.add_argument("--in", dest="core_path", required=False, help="Core parquet to evaluate on (optional if using --year)")
    pe.add_argument("--year", type=int, default=None, help="Shortcut for --in <data-dir>/core_YYYY.parquet")
    pe.add_argument("--data-dir", default="data", help="Base data directory for --year shortcut (default: data/)")

    pe.add_argument("--out", default=None, help="Optional output dir (default: run dir)")
    pe.add_argument("--threshold-key", default=None, help="Which threshold key to use (default: model name from manifest)")
    pe.add_argument("--weight-col", default="WTFA_A")
    pe.set_defaults(func=lambda a: evaluate_mod.cli(_rebuild_argv("evaluate", a)))

    # subgroup
    ps = sub.add_parser("subgroup", help="Subgroup evaluation (sex/age/education + raw columns)")
    ps.add_argument("--run", required=False, help="Run directory or manifest.json path")
    ps.add_argument("--task", default=None, help="Task name (used with --latest)")
    ps.add_argument("--latest", action="store_true", help="Use latest run for the given --task")
    ps.add_argument("--runs-dir", default="runs", help="Base runs directory (default: runs/)")

    ps.add_argument("--in", dest="core_path", required=False, help="Core parquet to evaluate on (optional if using --year)")
    ps.add_argument("--year", type=int, default=None, help="Shortcut for --in <data-dir>/core_YYYY.parquet")
    ps.add_argument("--data-dir", default="data", help="Base data directory for --year shortcut (default: data/)")

    ps.add_argument("--out", default=None, help="Optional output csv path (default: <run_dir>/subgroups_task=<task>.csv)")
    ps.add_argument("--by", nargs="+", required=True, help="Subgroups: sex age education or raw columns like REGION EDUCP_A")
    ps.add_argument("--weight-col", default="WTFA_A")
    ps.add_argument("--threshold-key", default=None, help="Threshold key (default: manifest['model'])")
    ps.add_argument("--min-n", type=int, default=200)
    ps.add_argument("--min-pos", type=int, default=25)
    ps.add_argument("--min-neg", type=int, default=25)

    # manual argv builder
    ps.set_defaults(func=lambda a: subgroup_mod.cli(_subgroup_argv(a)))

    # list-tasks
    plt = sub.add_parser("list-tasks", help="List available prediction tasks")
    plt.set_defaults(func=_cmd_list_tasks)

    # describe-task
    pdt = sub.add_parser("describe-task", help="Describe a task (required cols, description)")
    pdt.add_argument("task", help="Task name")
    pdt.set_defaults(func=_cmd_describe_task)

    # list-featuresets
    plf = sub.add_parser("list-featuresets", help="List available feature sets")
    plf.set_defaults(func=_cmd_list_featuresets)

    # describe-featureset
    pdfs = sub.add_parser("describe-featureset", help="Describe a feature set (counts and column lists)")
    pdfs.add_argument("featureset", help="Feature set name")
    pdfs.set_defaults(func=_cmd_describe_featureset)

    args = p.parse_args(argv)
    args.func(args)


def _cmd_list_tasks(args: argparse.Namespace) -> None:
    for t in list_tasks():
        print(t)


def _cmd_describe_task(args: argparse.Namespace) -> None:
    t = make_task(args.task)
    print(f"name: {t.name}")
    print(f"type: {t.problem_type}")
    print(f"description: {t.description}")
    print(f"required_cols: {t.required_cols}")


def _cmd_list_featuresets(args: argparse.Namespace) -> None:
    for f in list_featuresets():
        print(f)


def _cmd_describe_featureset(args: argparse.Namespace) -> None:
    fs = get_featureset(args.featureset)
    print(f"name: {fs.name}")
    if hasattr(fs, "description"):
        print(f"description: {getattr(fs, 'description')}")
    print(f"n_binary_12: {len(fs.binary_12)}")
    print(f"n_ordinal: {len(fs.ordinal)}")
    print(f"n_categorical: {len(fs.categorical)}")
    print("binary_12:", ", ".join(fs.binary_12))
    print("ordinal:", ", ".join(fs.ordinal))
    print("categorical:", ", ".join(fs.categorical))


def _rebuild_argv(cmd: str, args: argparse.Namespace) -> list[str]:
    flag_map = {
        "core_path": "--in",
        "run_dir": "--run-dir",
        "runs_dir": "--runs-dir",
        "weight_col": "--weight-col",
        "threshold_key": "--threshold-key",
        "data_dir": "--data-dir",
        "out_dir": "--out-dir",
    }

    d = vars(args).copy()
    d.pop("cmd", None)
    d.pop("func", None)

    out: list[str] = []
    for k, v in d.items():
        if v is None:
            continue

        flag = flag_map.get(k, "--" + k.replace("_", "-"))

        if isinstance(v, bool):
            if v:
                out.append(flag)
            continue

        if isinstance(v, list):
            for item in v:
                out.extend([flag, str(item)])
            continue

        out.extend([flag, str(v)])

    return out


def _subgroup_argv(args: argparse.Namespace) -> list[str]:
    out: list[str] = []

    if args.run is not None:
        out += ["--run", str(args.run)]
    if args.task is not None:
        out += ["--task", str(args.task)]
    if args.latest:
        out += ["--latest"]
    if args.runs_dir is not None:
        out += ["--runs-dir", str(args.runs_dir)]

    if args.core_path is not None:
        out += ["--in", str(args.core_path)]
    if args.year is not None:
        out += ["--year", str(args.year)]
    if args.data_dir is not None:
        out += ["--data-dir", str(args.data_dir)]

    out += ["--by", *list(args.by)]

    if args.out is not None:
        out += ["--out", str(args.out)]
    if args.weight_col is not None:
        out += ["--weight-col", str(args.weight_col)]
    if args.threshold_key is not None:
        out += ["--threshold-key", str(args.threshold_key)]

    out += ["--min-n", str(args.min_n), "--min-pos", str(args.min_pos), "--min-neg", str(args.min_neg)]
    return out
