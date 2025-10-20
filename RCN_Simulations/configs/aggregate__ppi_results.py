# RCN_Simulations/configs/aggregate_ppi_results.py
import os, glob
import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTIAL_DIR = os.path.join(REPO_DIR, "RCN_Simulations", "logs", "PPI", "partial")
FINAL_CSV  = os.path.join(REPO_DIR, "RCN_Simulations", "logs", "PPI", "PPI_Sweep_results.csv")

os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
paths = sorted(glob.glob(os.path.join(PARTIAL_DIR, "run_*.csv")))
if not paths:
    print(f"[WARN] No partial CSVs found under {PARTIAL_DIR}")
    raise SystemExit(0)

dfs = []
for p in paths:
    try:
        dfs.append(pd.read_csv(p))
    except Exception:
        print(f"[SKIP] failed to read {p}")

out = pd.concat(dfs, ignore_index=True, sort=False)
out.to_csv(FINAL_CSV, index=False)
print(f"[OK] Merged {len(dfs)} files -> {FINAL_CSV}")
