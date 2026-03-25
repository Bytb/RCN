#!/usr/bin/env python3
# RCN_Simulations/Cycle_Diagnostics.py
"""
Cycle_Diagnostics.py

Edge-attribute diagnostics for PPI gpickle to locate why "full RNBRW" weights are all-ones.

What it does:
  1) Loads the SAME gpickle used by your loaders: Data/Graphs/ppi_graph.gpickle
  2) Iterates over G.edges(data=True) and:
      - counts attribute-key frequencies across edges
      - checks candidate weight keys: ["weight", "ret_n", "ret", "rnbrw", "rnbrw_weight"]
      - reports missing rates per key
      - reports min/mean/max/std + approx-unique-count for each key (on non-missing edges)
      - reports whether a key is effectively constant (std==0 or unique_count==1)
  3) Calls load_ppi_cp_graph(use_onehot=True) and prints stats for the actual edge_weight tensor
     returned by the loader (this is what your "full" training currently uses).

Usage:
  python -m RCN_Simulations.PPI_Cycle_Diagnostics
  python -m RCN_Simulations.PPI_Cycle_Diagnostics --out ./attr_diag.txt
"""

from __future__ import annotations

import os
import sys
import math
import pickle
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# --- project imports ---
from Data.PPI import load_ppi_cp_graph


# ------------------------
# Helpers
# ------------------------
def _stats(vec: np.ndarray) -> Tuple[float, float, float, float]:
    v = np.asarray(vec, dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return (float(np.min(v)), float(np.mean(v)), float(np.max(v)), float(np.std(v)))


def _unique_count_approx(vec: np.ndarray, decimals: int = 8) -> int:
    v = np.asarray(vec, dtype=np.float64)
    if v.size == 0:
        return 0
    return int(np.unique(np.round(v, decimals)).size)


def _sha256_f32(vec: np.ndarray) -> str:
    v = np.asarray(vec, dtype=np.float32)
    h = hashlib.sha256()
    h.update(v.tobytes())
    return h.hexdigest()


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        # handle numpy scalars
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        # handle torch scalar
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _default_out_path(repo_dir: str) -> str:
    out_dir = os.path.join(repo_dir, "RCN_Simulations", "logs", "PPI_CP_OverlapPoC_Baselines", "partial")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "edge_attr_diagnostics.txt")


def _get_out_path(repo_dir: str) -> str:
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        return os.path.abspath(sys.argv[i + 1])
    return _default_out_path(repo_dir)


# ------------------------
# Main
# ------------------------
def main() -> None:
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = _get_out_path(repo_dir)

    # Paths that match your loader
    data_dir = os.path.join(repo_dir, "Data", "Graphs")
    ppi_path = os.path.join(data_dir, "ppi_graph.gpickle")

    candidate_keys = ["weight", "ret_n", "ret", "rnbrw", "rnbrw_weight", "cycle", "retrace"]

    lines: List[str] = []
    lines.append("PPI Edge Attribute Diagnostics\n")
    lines.append(f"repo_dir={repo_dir}\n")
    lines.append(f"ppi_path={ppi_path}\n")
    lines.append(f"candidate_keys={candidate_keys}\n\n")

    # ------------------------
    # 1) Inspect raw gpickle edge attributes
    # ------------------------
    if not os.path.exists(ppi_path):
        raise FileNotFoundError(f"Missing gpickle at: {ppi_path}")

    with open(ppi_path, "rb") as f:
        G = pickle.load(f)

    num_edges = G.number_of_edges()
    lines.append(f"[GPK] nodes={G.number_of_nodes()} edges={num_edges}\n")

    key_counter = Counter()
    missing_counter = Counter()
    nonfloat_counter = Counter()

    values_by_key: Dict[str, List[float]] = {k: [] for k in candidate_keys}

    # Sample a few edge-attr dicts for debugging
    sample_edges = []
    sample_limit = 5

    for (u, v, attrs) in G.edges(data=True):
        if len(sample_edges) < sample_limit:
            sample_edges.append((u, v, dict(attrs)))

        # Count keys present
        for k in attrs.keys():
            key_counter[k] += 1

        # Pull candidate keys
        for k in candidate_keys:
            if k not in attrs:
                missing_counter[k] += 1
                continue
            val = _safe_float(attrs.get(k))
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                nonfloat_counter[k] += 1
                continue
            values_by_key[k].append(val)

    lines.append("\n[GPK] Sample edge attribute dicts (first 5):\n")
    for u, v, attrs in sample_edges:
        # Keep it readable: show keys + a few key-values
        keys = sorted(list(attrs.keys()))
        preview = {kk: attrs[kk] for kk in keys[:10]}
        lines.append(f"  edge({u}, {v}) keys={keys}\n")
        lines.append(f"    preview_first10={preview}\n")

    lines.append("\n[GPK] Most common attribute keys (top 30):\n")
    for k, c in key_counter.most_common(30):
        lines.append(f"  {k}: {c}\n")

    lines.append("\n[GPK] Candidate-key coverage + stats (computed over non-missing, numeric values):\n")
    for k in candidate_keys:
        present_n = len(values_by_key[k])
        missing_n = missing_counter[k]
        nonfloat_n = nonfloat_counter[k]
        coverage = 100.0 * present_n / max(1, num_edges)

        mn, mu, mx, sd = _stats(np.array(values_by_key[k], dtype=np.float64))
        uc = _unique_count_approx(np.array(values_by_key[k], dtype=np.float64), decimals=8)
        sha = _sha256_f32(np.array(values_by_key[k], dtype=np.float64)) if present_n > 0 else "NA"

        lines.append(f"\n  key='{k}'\n")
        lines.append(f"    present_numeric={present_n} / {num_edges} ({coverage:.3f}%)\n")
        lines.append(f"    missing={missing_n}, non_numeric_or_nan={nonfloat_n}\n")
        lines.append(f"    min={mn:.6g} mean={mu:.6g} max={mx:.6g} std={sd:.6g}\n")
        lines.append(f"    unique_count_approx={uc}\n")
        lines.append(f"    sha256_f32(nonmissing_only)={sha}\n")

        if present_n > 0 and (uc <= 1 or sd == 0.0):
            lines.append("    NOTE: looks constant/degenerate on present edges.\n")

    # ------------------------
    # 2) Compare to what your loader actually returns as edge_weight
    # ------------------------
    lines.append("\n\n[LOADER] load_ppi_cp_graph(...) edge_weight diagnostics (this is what 'full' uses):\n")
    x, edge_index, y, edge_weight, G2, communities, y_multi, mask, nid2idx, idx2nid = load_ppi_cp_graph(
        use_onehot=True, device="cpu"
    )

    ew = edge_weight.detach().cpu().numpy().astype(np.float64)
    mn, mu, mx, sd = _stats(ew)
    uc = _unique_count_approx(ew, decimals=8)
    sha = _sha256_f32(ew)

    lines.append(f"  edge_index_cols={edge_index.shape[1]}\n")
    lines.append(f"  edge_weight_len={ew.size}\n")
    lines.append(f"  min={mn:.6g} mean={mu:.6g} max={mx:.6g} std={sd:.6g}\n")
    lines.append(f"  unique_count_approx={uc}\n")
    lines.append(f"  sha256_f32={sha}\n")
    lines.append(f"  first20={np.array2string(ew[:20], precision=6, separator=', ')}\n")

    if uc <= 1 or sd == 0.0:
        lines.append("  NOTE: loader edge_weight appears constant/degenerate.\n")

    # Write report
    with open(out_path, "w") as f:
        f.writelines(lines)

    print(f"[OK] Wrote {out_path}")

    # Hard fail if loader weights are degenerate (this matches your professor's intent)
    if not (sd > 0.0 and uc > 1):
        raise AssertionError(
            f"[ASSERT FAIL] Loader edge_weight is degenerate: std={sd:.6g}, unique_count={uc}. "
            f"This implies 'full' is effectively unweighted."
        )


if __name__ == "__main__":
    main()
    os._exit(0)
