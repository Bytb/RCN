#!/usr/bin/env python3
"""
combine_results.py

Collects all run_*.csv from logs/PPI/partial/ and vertically concatenates them
into logs/PPI/combined_hypersweep_results.csv (default names/paths).
Designed to live in RCN_Simulations/ next to ArrayRunner.py.

Usage:
    python combine_results.py
    python combine_results.py --base-dir /path/to/RCN_Simulations
    python combine_results.py --no-source-file
    python combine_results.py --outfile combined_hypersweep_results.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import re
import csv
import pandas as pd

def detect_sep(sample_path: Path) -> str:
    """
    Try to detect delimiter (tab vs comma) using csv.Sniffer on a small sample.
    Fallback to comma if uncertain.
    """
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
        return ","  # safe default

def read_csv_robust(path: Path, sep: str | None):
    """
    Read a CSV robustly:
      - try utf-8, fallback latin-1
      - if sep is None, let pandas infer (engine='python')
    """
    for enc in ("utf-8", "latin-1"):
        try:
            if sep is None:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc)
            else:
                return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            raise
    # If we somehow get here:
    return pd.read_csv(path, sep=sep or ",", encoding="latin-1", engine="python")

def parse_run_number(filename: str) -> int:
    """
    Extract numeric part from 'run_XXXX.csv' (e.g., run_0007.csv -> 7).
    Unparseable -> +inf to sink to end but shouldn’t happen with your pattern.
    """
    m = re.search(r"run_(\d+)\.csv$", filename)
    if m:
        return int(m.group(1))
    return 10**12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,  # RCN_Simulations by default
        help="Directory that contains logs/PPI/partial/ (default: script folder).",
    )
    ap.add_argument(
        "--partial-subpath",
        type=Path,
        default=Path("logs") / "PPI_CP_Sweep2" / "partial",
        help="Relative path to partial CSVs under base-dir.",
    )
    ap.add_argument(
        "--out-dir-subpath",
        type=Path,
        default=Path("logs") / "PPI_CP_Sweep2",
        help="Where to write the combined CSV (relative to base-dir).",
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="combined_hypersweep_results_CP_Sweep2.csv",
        help="Output filename (will be created in out-dir-subpath).",
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
    out_path = out_dir / args.outfile

    if not partial_dir.exists():
        print(f"[ERROR] Partial directory not found: {partial_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        (p for p in partial_dir.glob("run_*.csv") if p.is_file()),
        key=lambda p: parse_run_number(p.name),
    )

    if not files:
        print(f"[WARN] No files matched {partial_dir / 'run_*.csv'}. Nothing to do.")
        sys.exit(0)

    # Detect delimiter from the first file (helps if your headers are tab-separated)
    sep = detect_sep(files[0])

    dfs = []
    skipped = []
    first_cols = None
    for p in files:
        try:
            df = read_csv_robust(p, sep=sep if sep else None)
            # Ensure non-empty and has columns
            if df.empty and len(df.columns) == 0:
                skipped.append((p.name, "empty/no columns"))
                continue
            if not args.no_source_file:
                df.insert(0, "SourceFile", p.name)
            if first_cols is None:
                first_cols = list(df.columns)
            dfs.append(df)
        except Exception as e:
            skipped.append((p.name, f"read error: {e!r}"))

    if not dfs:
        print("[ERROR] No valid CSVs to combine after reading attempts.", file=sys.stderr)
        for name, reason in skipped:
            print(f"  - Skipped {name}: {reason}")
        sys.exit(2)

    # Concatenate without sorting to preserve the column order from the first valid file.
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # Put columns in a stable order: first the first-file columns (if known), then any extras.
    if first_cols is not None:
        extra_cols = [c for c in combined.columns if c not in first_cols]
        combined = combined[first_cols + extra_cols]

    combined.to_csv(out_path, index=False)

    print(f"[OK] Combined {len(dfs)} files into: {out_path}")
    print(f"     Total rows: {len(combined):,}")
    if skipped:
        print("     Skipped files:")
        for name, reason in skipped:
            print(f"       - {name}: {reason}")

if __name__ == "__main__":
    main()
