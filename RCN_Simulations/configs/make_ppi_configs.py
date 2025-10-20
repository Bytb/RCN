# RCN_Simulations/configs/make_ppi_configs.py
import os, csv
from itertools import product

# --- Config grid (matches what you specified) ---
SEEDS = [120, 328476, 42]
lambda_mods      = [0.3, 0.6, 0.9]
lambda_laps      = [0.001, 0.0001, 1e-5]
lambda_contrasts = [0.001, 0.0001, 1e-5]
lambda_orth      = [0.001, 0.0001, 1e-5, 1e-6]

rows = []
for lm, ll, lc, lo, s in product(lambda_mods, lambda_laps, lambda_contrasts, lambda_orth, SEEDS):
    rows.append({"Seed": s, "lambda_mod": lm, "lambda_lap": ll, "lambda_contrast": lc, "lambda_orth": lo})

# --- Write CSV ---
cfg_dir = os.path.join(os.path.dirname(__file__))
out_csv = os.path.join(cfg_dir, "PPI_configs.csv")
os.makedirs(cfg_dir, exist_ok=True)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Seed", "lambda_mod", "lambda_lap", "lambda_contrast", "lambda_orth"])
    w.writeheader()
    w.writerows(rows)

print(f"[OK] Wrote {len(rows)} configs -> {out_csv}")
