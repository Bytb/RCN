# RCN_Simulations/HyperParam_ArrayRunner.py
import os, sys, math, traceback
import random
import numpy as np
import pandas as pd
import torch

# --- project imports (unchanged) ---
from Models.RCN import CAGATAblationModel
from Models.LossFunctions import combined_community_loss_PPI
from HelperFunctions import cluster_with_kmeans, onmi_mgh
from Data.PPI import load_ppi_cp_graph  # your PPI loader
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ------------------------
# Constants / Paths
# ------------------------
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_CSV = os.path.join(os.path.dirname(__file__), "configs", "PPI_sweep2_configs.csv")

DATASET = "PPI"
K_FIXED = 793  # fixed number of clusters for k-means (same as your script)
EPOCHS  = 200   # ← per your request
LR      = 0.005

# IMPORTANT: When you're ready to use K=1000 negatives, expose it in your loss and pass here.
# e.g., combined_community_loss_PPI(..., contrast_K=1000)
CONTRAST_TAU = 0.5
CONTRAST_VARIANT = "node"

# ------------------------
# Utilities
# ------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def _read_config_row(index: int):
    df = pd.read_csv(CONFIG_CSV)
    if index < 0 or index >= len(df):
        raise IndexError(f"Config index {index} out of range [0, {len(df)-1}]")
    r = df.iloc[index]
    return int(r["Seed"]), float(r["lambda_mod"]), float(r["lambda_lap"]), float(r["lambda_contrast"]), float(r["lambda_orth"])

def _slurm_index_or_arg():
    # Prefer SLURM_ARRAY_TASK_ID; fallback to CLI: --index N
    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is not None:
        return int(env_idx)
    # CLI fallback: python -m RCN_Simulations.PPI_HyperParam_ArrayRunner --index 123
    if "--index" in sys.argv:
        i = sys.argv.index("--index")
        return int(sys.argv[i+1])
    raise RuntimeError("No SLURM_ARRAY_TASK_ID; run with --index N for local testing")

def _ensure_dirs():
    out_dir = os.path.join(REPO_DIR, "RCN_Simulations", "logs", "PPI_CP_Sweep2", "partial")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ------------------------
# Main single-config run
# ------------------------
def main():
    out_dir = _ensure_dirs()
    idx = _slurm_index_or_arg()
    seed, lm, ll, lc, lo = _read_config_row(idx)

    # Per-run outputs
    csv_path   = os.path.join(out_dir, f"run_{idx:04d}.csv")
    error_path = os.path.join(out_dir, f"run_{idx:04d}_error.txt")
 
    # --- Skip if already completed successfully ---

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if len(df) >= 1:
                r = df.iloc[0]
                reason = str(r.get("ONMI_reason", "")).strip()
                sil = r.get("Silhouette", float("nan"))
                already_ok = (reason == "" or reason.lower() == "nan") and (not pd.isna(sil))
                if already_ok:
                    print(f"[SKIP] index={idx} already completed -> {csv_path}")
                    return
        except Exception:
            pass

    print(f"[RUN] index={idx} seed={seed} λ_mod={lm} λ_lap={ll} λ_contrast={lc} λ_orth={lo}")
    sys.stdout.flush()

    # Device (CPU-only by your plan; keep CUDA check harmless)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # --- Reproducibility ---
        set_seed(seed)

        # --- Load data ---
        x, edge_index, y, edge_weight, G, communities, y_multi, mask, nid2idx, idx2nid = load_ppi_cp_graph(use_onehot=True, device=device)
        x, edge_index, y, edge_weight = x.to(device), edge_index.to(device), y.to(device), edge_weight.to(device)
        
        # >>> PATCH 1 START (GT community diagnostics) <<<
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask, dtype=bool)
         
        U = set(np.nonzero(mask_np)[0].tolist())
        valid_nodes = sorted(U)
         
        # raw survivors (no dedup), requiring len>=2
        survivors_raw = sum(1 for c in communities if len(set(c) & U) >= 2)
         
        # deduplicated survivors (what usually matches ~1995), requiring len>=2
        gt_masked_dedup = set()
        for c in communities:
            cc = frozenset(set(c) & U)
            if len(cc) >= 2:
                gt_masked_dedup.add(cc)
         
        print(f"[DEBUG] GT survivors (raw, >=2): {survivors_raw}")
        print(f"[DEBUG] GT survivors (dedup, >=2): {len(gt_masked_dedup)}  <-- this should be ~1995")
        # >>> PATCH 1 END <<<

        # --- Build valid node mask and GT diagnostics (same as your script) ---
        #if isinstance(mask, torch.Tensor):
        #    mask_np = mask.detach().cpu().numpy().astype(bool)
        #else:
        #    mask_np = np.asarray(mask, dtype=bool)

        #valid_nodes = np.nonzero(mask_np)[0].tolist()
        #gt_masked = []
        #for c in communities:
        #    cc = set(c) & set(valid_nodes)
        #    if len(cc) >= 2:
        #        gt_masked.append(cc)

        # --- Model ---
        model = CAGATAblationModel(
            in_dim=x.size(1),
            hidden_dim=8,
            out_dim=K_FIXED,
            heads=4
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)

        # --- Train (EPOCHS=200) ---
        model.train()
        last_logs = None
        for epoch in range(EPOCHS):
            optim.zero_grad()
            _, _, embeddings = model(x, edge_index, edge_weight)

            # e.g., combined_community_loss_PPI(..., contrast_tau=CONTRAST_TAU, contrast_variant=CONTRAST_VARIANT, contrast_K=1000, return_stats=True)
            loss, logs = combined_community_loss_PPI(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_weight=edge_weight,
                lambda_mod=lm,
                lambda_lap=ll,
                lambda_contrast=lc,
                lambda_orth=lo,
                contrast_tau=CONTRAST_TAU,
                contrast_variant=CONTRAST_VARIANT,
                return_stats=True
            )
            last_logs = logs
            loss.backward()
            optim.step()

        # --- Final embeddings ---
        model.eval()
        with torch.no_grad():
            _, _, embeddings = model(x, edge_index, edge_weight)

        # --- Clustering & metrics (same schema as your script) ---
        preds = cluster_with_kmeans(embeddings, K_FIXED, seed=seed)
        Z = to_numpy(embeddings)

        try:  sil = float(silhouette_score(Z, preds))
        except Exception: sil = float("nan")
        try:  ch = float(calinski_harabasz_score(Z, preds))
        except Exception: ch = float("nan")
        try:  db = float(davies_bouldin_score(Z, preds))
        except Exception: db = float("nan")

        # Pred communities for ONMI
        pred_map = {}
        for idx_node, lab in enumerate(to_numpy(preds)):
            pred_map.setdefault(int(lab), set()).add(int(idx_node))
        pred_comms = [pred_map[k] for k in sorted(pred_map.keys())]

        pred_masked = []
        for c in pred_comms:
            cc = set(c) & set(valid_nodes)
            if len(cc) > 0:
                pred_masked.append(cc)

        U = set(valid_nodes)
        gt_sets = []
        for c in communities:
            cc = set(c) & U
            if len(cc) > 0:
                gt_sets.append(frozenset(cc))
        onmi_style_gt_count = len(set(gt_sets))

        pred_sets = []
        for c in pred_comms:
            cc = set(c) & U
            if len(cc) > 0:
                pred_sets.append(frozenset(cc))
        onmi_style_pred_count = len(set(pred_sets))

        onmi_reason = ""
        if len(valid_nodes) < 2:
            onmi = float("nan"); onmi_reason = "no_masked_nodes"
        elif len(gt_masked_dedup) == 0:
            onmi = float("nan"); onmi_reason = "empty_gt"
        elif len(pred_masked) == 0:
            onmi = float("nan"); onmi_reason = "empty_pred"
        else:
            onmi = float(onmi_mgh(communities, pred_comms, nodes_mask=valid_nodes))
            if np.isnan(onmi):
                onmi_reason = "zero_entropy_or_degenerate"

        extra_logs = {}
        if last_logs is not None:
            extra_logs = {
                "TrainLoss_Final": round(last_logs.get("loss_total", float("nan")), 6),
                "Contrast_TotalW": round(last_logs.get("contrast_total_w", 0.0), 3),
                "Contrast_PctValid": round(100.0 * last_logs.get("contrast_pct_valid", 0.0), 2),
                "Contrast_PosPerRow": round(last_logs.get("contrast_mean_pos_per_row", 0.0), 4),
                "Contrast_MeanLSE_Pos": round(last_logs.get("contrast_mean_lse_pos", 0.0), 4),
                "Contrast_MeanLSE_Neg": round(last_logs.get("contrast_mean_lse_neg", 0.0), 4),
            }

        row = {
            "Graph": DATASET,
            "Seed": int(seed),
            "K": int(K_FIXED),
            "λ_mod": float(lm),
            "λ_lap": float(ll),
            "λ_contrast": float(lc),
            "λ_orth": float(lo),
            "Silhouette": (None if np.isnan(sil) else round(sil, 4)),
            "CalinskiHarabasz": (None if np.isnan(ch) else round(ch, 4)),
            "DaviesBouldin": (None if np.isnan(db) else round(db, 4)),
            "ONMI": (None if np.isnan(onmi) else round(onmi, 4)),
            "ONMI_valid_nodes": int(len(valid_nodes)),
            "ONMI_GT_comms": int(onmi_style_gt_count),
            "ONMI_Pred_comms": int(onmi_style_pred_count),
            "ONMI_reason": onmi_reason,
            **extra_logs,
        }

        pd.DataFrame([row]).to_csv(csv_path, index=False)
        print(f"[OK] Wrote {csv_path}")

    except Exception as e:
        # Write error + NaN row so aggregation still works
        err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        with open(error_path, "w") as f:
            f.write(f"[ERROR] index={idx} seed={seed} lm={lm} ll={ll} lc={lc} lo={lo}\n")
            f.write(err_msg)
        nan_row = {
            "Graph": DATASET, "Seed": int(seed), "K": int(K_FIXED),
            "λ_mod": float(lm), "λ_lap": float(ll), "λ_contrast": float(lc), "λ_orth": float(lo),
            "Silhouette": None, "CalinskiHarabasz": None, "DaviesBouldin": None,
            "ONMI": None, "ONMI_valid_nodes": 0, "ONMI_GT_comms": 0, "ONMI_Pred_comms": 0,
            "ONMI_reason": f"EXCEPTION: {type(e).__name__}",
        }
        pd.DataFrame([nan_row]).to_csv(csv_path, index=False)
        print(f"[FAIL] index={idx} -> wrote NaN CSV and error log at {error_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    os._exit(0)
