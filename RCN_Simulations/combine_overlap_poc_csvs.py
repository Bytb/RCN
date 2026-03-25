#!/usr/bin/env python3
"""
combine_overlap_poc_csvs.py

Combines two groups of CSVs produced by the PPI_CP_OverlapPoC experiments:

1) Overlap PoC runs:
   logs/PPI_CP_OverlapPoC/partial/run_*.csv
   -> logs/PPI_CP_OverlapPoC/combined_overlap_runs.csv

2) K-sweep runs:
   logs/PPI_CP_OverlapPoC/partial/k_sweep_seed*.csv
   -> logs/PPI_CP_OverlapPoC/combined_k_sweep.csv

Also generates Step-2 K-sweep figure (averaged across seeds):
- ONMI vs K (r=1 and best-r)
- # predicted communities surviving >=2 vs K
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def parse_run_number(filename: str) -> int:
    """Extract numeric part from run_XXXX.csv; unparseable -> big number."""
    m = re.search(r"run_(\d+)\.csv$", filename)
    if m:
        return int(m.group(1))
    return 10**12


def combine_group(
    files: list[Path],
    out_path: Path,
    add_source_file: bool = True,
) -> None:
    """Combine a list of CSV files into out_path."""
    if not files:
        print(f"[WARN] No input files for {out_path.name}. Nothing to do.")
        return

    # Detect delimiter from the first file
    sep = detect_sep(files[0])

    dfs = []
    skipped: list[tuple[str, str]] = []
    first_cols: list[str] | None = None

    for p in files:
        try:
            df = read_csv_robust(p, sep=sep if sep else None)

            # Ensure non-empty and has columns
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


# -------------------------
# Plotting (Step 2)
# -------------------------
def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_k_sweep(ksweep_csv: Path, out_dir: Path) -> Path | None:
    """
    Create one figure:
      - ONMI vs K (ONMI_r1 and Best_ONMI_r1to5) averaged across seeds
      - # predicted communities with >=2 (PredComms_GE2_at_Best) averaged across seeds
    """
    if not ksweep_csv.exists():
        print(f"[WARN] K-sweep combined CSV not found: {ksweep_csv}. Skipping plot.")
        return None

    df = pd.read_csv(ksweep_csv)
    needed = ["K", "ONMI_r1", "Best_ONMI_r1to5", "PredComms_GE2_at_Best"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in {ksweep_csv.name}: {missing}. Skipping plot.")
        return None

    df = _coerce_numeric(df, needed)

    # Average across seeds for each K
    g = (
        df.groupby("K", as_index=False)[["ONMI_r1", "Best_ONMI_r1to5", "PredComms_GE2_at_Best"]]
        .mean(numeric_only=True)
        .sort_values("K")
    )

    # Drop rows where K is NaN or no metrics
    g = g.dropna(subset=["K"])

    if g.empty:
        print(f"[WARN] No usable rows after grouping in {ksweep_csv.name}. Skipping plot.")
        return None

    K = g["K"].astype(int).to_numpy()
    onmi_r1 = g["ONMI_r1"].to_numpy()
    onmi_best = g["Best_ONMI_r1to5"].to_numpy()
    pred_ge2 = g["PredComms_GE2_at_Best"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 4.8))

    # ONMI curves (left axis)
    ax1.plot(K, onmi_r1, marker="o", linestyle="-", label="ONMI (r=1)")
    ax1.plot(K, onmi_best, marker="o", linestyle="--", label="ONMI (best r=1..5)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("ONMI")
    ax1.set_xticks(K)
    ax1.grid(True, which="both", axis="both", linestyle=":")

    # Predicted communities (right axis)
    ax2 = ax1.twinx()
    ax2.plot(K, pred_ge2, marker="s", linestyle="-", label="#PredComms (>=2)")
    ax2.set_ylabel("# Predicted Communities (>=2)")

    # Single combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True)

    ax1.set_title("K-Sweep Sensitivity (Averaged Across Seeds)")

    out_path = out_dir / "k_sweep_onmi_and_predcomms_vs_k.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Wrote plot: {out_path}")
    return out_path


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,  # RCN_Simulations by default
        help="Directory that contains logs/PPI_CP_OverlapPoC/ (default: this script folder).",
    )
    ap.add_argument(
        "--partial-subpath",
        type=Path,
        default=Path("logs") / "PPI_CP_OverlapPoC" / "partial",
        help="Relative path to partial CSVs under base-dir.",
    )
    ap.add_argument(
        "--out-dir-subpath",
        type=Path,
        default=Path("logs") / "PPI_CP_OverlapPoC",
        help="Where to write combined CSVs (relative to base-dir).",
    )
    ap.add_argument(
        "--overlap-outfile",
        type=str,
        default="combined_overlap_runs.csv",
        help="Output filename for combined run_*.csv.",
    )
    ap.add_argument(
        "--ksweep-outfile",
        type=str,
        default="combined_k_sweep.csv",
        help="Output filename for combined k_sweep_seed*.csv.",
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

    # Group 1: overlap PoC run_*.csv
    run_files = sorted(
        (p for p in partial_dir.glob("run_*.csv") if p.is_file()),
        key=lambda p: parse_run_number(p.name),
    )

    # Group 2: K-sweep seed files
    ksweep_files = sorted(
        (p for p in partial_dir.glob("k_sweep_seed*.csv") if p.is_file()),
        key=lambda p: p.name,
    )

    overlap_out = out_dir / args.overlap_outfile
    ksweep_out = out_dir / args.ksweep_outfile

    combine_group(run_files, overlap_out, add_source_file=(not args.no_source_file))
    combine_group(ksweep_files, ksweep_out, add_source_file=(not args.no_source_file))

    # Step 2 figure (averaged across seeds)
    plot_k_sweep(ksweep_out, out_dir)


if __name__ == "__main__":
    main()
