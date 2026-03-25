#!/usr/bin/env python3
# RCN_Simulations/combine_full_vs_cyclebreak.py
"""
Combine the 4 per-run CSVs from:
  RCN_Simulations/logs/PPI_CP_FullVsCyclebreak/partial/

and produce:
  - combined_runs.csv  (all rows concatenated)
  - delta_table.csv    (per seed: ONMI(full), ONMI(cyclebreak), Delta)

Usage:
  python combine_full_vs_cyclebreak.py
  python combine_full_vs_cyclebreak.py --base-dir /path/to/RCN_Simulations
  python combine_full_vs_cyclebreak.py --outdir /path/to/output_folder
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,  # RCN_Simulations by default
        help="Directory containing logs/PPI_CP_FullVsCyclebreak/partial/ (default: this script folder).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: logs/PPI_CP_FullVsCyclebreak/).",
    )
    args = ap.parse_args()

    base = args.base_dir.resolve()
    partial_dir = base / "logs" / "PPI_CP_FullVsCyclebreak" / "partial"
    if not partial_dir.exists():
        raise FileNotFoundError(f"Missing partial dir: {partial_dir}")

    outdir = args.outdir.resolve() if args.outdir is not None else (base / "logs" / "PPI_CP_FullVsCyclebreak")
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect run_*.csv files
    files = sorted(partial_dir.glob("run_*_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"No run CSVs found in {partial_dir}")

    df_all = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["SourceFile"] = fp.name
            df_all.append(df)
        except Exception as e:
            print(f"[WARN] failed reading {fp}: {e}", file=sys.stderr)

    if not df_all:
        raise RuntimeError("No readable CSV files found.")

    combined = pd.concat(df_all, ignore_index=True)
    combined_path = outdir / "combined_runs.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[OK] wrote {combined_path}")

    # Build delta table: per seed (requires both variants)
    required_cols = {"Seed", "Variant", "ONMI_r2"}
    missing = required_cols - set(combined.columns)
    if missing:
        raise KeyError(f"combined_runs.csv missing required columns: {sorted(missing)}")

    # Pivot ONMI by variant
    pivot = combined.pivot_table(index="Seed", columns="Variant", values="ONMI_r2", aggfunc="first")
    # Ensure expected variants exist
    for v in ["full", "cyclebreak_shuffle"]:
        if v not in pivot.columns:
            pivot[v] = pd.NA

    delta = pd.DataFrame({
        "Seed": pivot.index.astype(int),
        "ONMI_full": pivot["full"],
        "ONMI_cyclebreak": pivot["cyclebreak_shuffle"],
    }).reset_index(drop=True)

    delta["Delta_full_minus_cyclebreak"] = delta["ONMI_full"] - delta["ONMI_cyclebreak"]

    delta_path = outdir / "delta_table.csv"
    delta.to_csv(delta_path, index=False)
    print(f"[OK] wrote {delta_path}")

    # Also print a tiny console view
    print("\nDelta table:")
    print(delta.to_string(index=False))


if __name__ == "__main__":
    main()
