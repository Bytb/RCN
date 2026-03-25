#!/usr/bin/env python3
"""
combine_cycle_baselines_csvs.py

Combines CSVs produced by the PPI-CP RCN mechanism controls (cycle baselines).

Inputs (default):
  logs/PPI_CP_OverlapPoC_Baselines/partial/mechanism_*_seed*.csv

Outputs (default):
  logs/PPI_CP_OverlapPoC_Baselines/combined_mechanism_controls_raw.csv
  logs/PPI_CP_OverlapPoC_Baselines/combined_mechanism_controls_summary.csv

Summary is grouped by (Variant, K) and reports mean/std across seeds.
No plots are generated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------
# CSV reading utilities
# -------------------------
def detect_sep(sample_path: Path) -> str:
    """Detect delimiter using csv.Sniffer; fallback to comma."""
    try:
        with open(sample_path, "r", encoding="utf-8", errors="strict") as f:
            sample = f.read(4096)
    except UnicodeDecodeError:
        with open(sample_path, "r", encoding="latin-1", errors="strict") as f:
            sample = f.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return ","


def read_csv_robust(path: Path, sep: str | None) -> pd.DataFrame:
    """Read CSV robustly with encoding fallback; sep=None lets pandas infer."""
    for enc in ("utf-8", "latin-1"):
        try:
            if sep is None:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc)
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=sep or ",", encoding="latin-1", engine="python")


def combine_group(files: list[Path], out_path: Path, add_source_file: bool = True) -> pd.DataFrame:
    """Combine a list of CSV files into out_path; returns combined DataFrame."""
    if not files:
        print(f"[WARN] No input files for {out_path.name}. Nothing to do.")
        return pd.DataFrame()

    sep = detect_sep(files[0])

    dfs = []
    skipped: list[tuple[str, str]] = []
    first_cols: list[str] | None = None

    for p in files:
        try:
            df = read_csv_robust(p, sep=sep if sep else None)

            if df.empty and len(df.columns) == 0:
                skipped.append((p.name, "empty/no columns"))
                continue

            if add_source_file:
                df.insert(0, "SourceFile", p.name)

            if first_cols is None:
                first_cols = list(df.columns)

            dfs.append(df)

        except Exception as e:
            skipped.append((p.name, f"read error: {e!r}"))

    if not dfs:
        print(f"[ERROR] No valid CSVs to combine for {out_path.name}.", file=sys.stderr)
        for name, reason in skipped:
            print(f"  - Skipped {name}: {reason}", file=sys.stderr)
        raise SystemExit(2)

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # Stable column order: first file columns + extras
    if first_cols is not None:
        extra_cols = [c for c in combined.columns if c not in first_cols]
        combined = combined[first_cols + extra_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"[OK] Wrote {out_path}")
    print(f"     Input files: {len(dfs)} | Rows: {len(combined):,}")
    if skipped:
        print("     Skipped files:")
        for name, reason in skipped:
            print(f"       - {name}: {reason}")

    return combined


# -------------------------
# Summary table
# -------------------------
def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_summary(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Group by (Variant, K) and compute mean/std across seeds.
    Also includes n_seeds (count of non-null Seed rows).
    """
    if df.empty:
        print(f"[WARN] Empty dataframe; skipping summary {out_path.name}.")
        return pd.DataFrame()

    # Expected columns from your mechanism runner
    required_any = ["Variant", "K"]
    missing = [c for c in required_any if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in combined raw CSV: {missing}")

    # Metrics to summarize if present
    metrics = [
        "ONMI_r2",
        "PredComms_GE2_r2",
        "Silhouette",
        "CH",
        "DB",
        "PredSize_Median_GE2_r2",
        "PredSize_P90_GE2_r2",
    ]
    present_metrics = [m for m in metrics if m in df.columns]

    # Ensure numeric where appropriate
    df = _coerce_numeric(df, ["K", "Seed"] + present_metrics)

    # Count seeds contributing per (Variant, K)
    # Use Seed if present, else count rows.
    if "Seed" in df.columns:
        n_col = df.groupby(["Variant", "K"])["Seed"].count().rename("n_seeds")
    else:
        n_col = df.groupby(["Variant", "K"]).size().rename("n_seeds")

    agg_dict = {}
    for m in present_metrics:
        agg_dict[m] = ["mean", "std"]

    g = df.groupby(["Variant", "K"], as_index=False).agg(agg_dict)

    # Flatten columns like ("ONMI_r2","mean") -> "ONMI_r2_mean"
    g.columns = [
        f"{a}_{b}" if b else a
        for (a, b) in ([(c[0], c[1] if len(c) > 1 else "") for c in g.columns.values])
    ]

    # Merge counts
    g = g.merge(n_col.reset_index(), on=["Variant", "K"], how="left")

    # Round for readability (keep K as int)
    if "K_" in g.columns:
        pass
    if "K" in g.columns:
        g["K"] = pd.to_numeric(g["K"], errors="coerce").astype("Int64")

    round_cols = [c for c in g.columns if c.endswith("_mean") or c.endswith("_std")]
    for c in round_cols:
        g[c] = g[c].astype(float).round(6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} | Rows: {len(g):,}")

    return g


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,  # RCN_Simulations by default
        help="Directory that contains logs/... (default: this script folder).",
    )
    ap.add_argument(
        "--partial-subpath",
        type=Path,
        default=Path("logs") / "PPI_CP_OverlapPoC_Baselines" / "partial",
        help="Relative path to partial CSVs under base-dir.",
    )
    ap.add_argument(
        "--out-dir-subpath",
        type=Path,
        default=Path("logs") / "PPI_CP_OverlapPoC_Baselines",
        help="Where to write combined CSVs (relative to base-dir).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="mechanism_*_seed*.csv",
        help="Glob pattern for partial CSVs inside partial-subpath.",
    )
    ap.add_argument(
        "--raw-outfile",
        type=str,
        default="combined_mechanism_controls_raw.csv",
        help="Output filename for combined raw CSV.",
    )
    ap.add_argument(
        "--summary-outfile",
        type=str,
        default="combined_mechanism_controls_summary.csv",
        help="Output filename for mean/std summary CSV.",
    )
    ap.add_argument(
        "--no-source-file",
        action="store_true",
        help="If set, do not add a SourceFile column.",
    )
    args = ap.parse_args()

    partial_dir = (args.base_dir / args.partial_subpath).resolve()
    out_dir = (args.base_dir / args.out_dir_subpath).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not partial_dir.exists():
        print(f"[ERROR] Partial directory not found: {partial_dir}", file=sys.stderr)
        raise SystemExit(1)

    files = sorted([p for p in partial_dir.glob(args.pattern) if p.is_file()], key=lambda p: p.name)
    if not files:
        print(f"[ERROR] No files matched pattern '{args.pattern}' in {partial_dir}", file=sys.stderr)
        raise SystemExit(2)

    raw_out = out_dir / args.raw_outfile
    summary_out = out_dir / args.summary_outfile

    combined = combine_group(files, raw_out, add_source_file=(not args.no_source_file))
    make_summary(combined, summary_out)


if __name__ == "__main__":
    main()
