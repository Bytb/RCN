#!/usr/bin/env python3
# RCN_Simulations/LambdaSensitivity_ArrayRunner.py
#
# 32-way SLURM array:
#   4 lambdas × 4 multipliers × 2 seeds = 32 tasks
# Each task trains once (200 epochs) and evaluates once (K=793, r=2),
# then writes ONE tiny CSV row to:
#   RCN_Simulations/logs/PPI_LambdaSensitivity/partial/
#
# Mapping (tid = SLURM_ARRAY_TASK_ID in 0..31):
#   lam_i  = tid // 8            -> 0..3
#   rem    = tid % 8
#   mult_i = rem // 2            -> 0..3
#   seed_i = rem % 2             -> 0..1
#
# Usage:
#   sbatch --array=0-31%32 ... python RCN_Simulations/LambdaSensitivity_ArrayRunner.py
#   (optional) python ... --index 7

import os, sys, traceback
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- project imports ---
from Models.RCN import CAGATAblationModel
from Models.LossFunctions import combined_community_loss_PPI
from HelperFunctions import onmi_mgh
from Data.PPI import load_ppi_cp_graph


# ------------------------
# Constants / Paths
# ------------------------
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OUT_FOLDER_NAME = "PPI_LambdaSensitivity"
EPOCHS = 200
LR = 0.005

# Fixed evaluation
K_FIXED = 793
R_FIXED = 2

CONTRAST_TAU = 0.5
CONTRAST_VARIANT = "node"

# Two seeds only (as requested)
SEEDS = [120, 42]

# Multipliers (as requested)
MULTIPLIERS = [0.0, 0.5, 1.0, 2.0]

# Base lambdas (provided)
FIXED_CFG = {"lambda_mod": 0.05, "lambda_lap": 1e-5, "lambda_contrast": 0.1, "lambda_orth": 1e-6}
LAMBDA_KEYS = ["lambda_mod", "lambda_lap", "lambda_contrast", "lambda_orth"]


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

def _ensure_dirs():
    out_dir = os.path.join(REPO_DIR, "RCN_Simulations", "logs", OUT_FOLDER_NAME, "partial")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _filter_comms_ge2(comms):
    """Keep only communities with >=2 members."""
    return [set(c) for c in comms if len(c) >= 2]

def _build_pred_comms_from_memberships(valid_nodes: np.ndarray, memberships: list[list[int]], K: int):
    cluster_to_nodes = [set() for _ in range(K)]
    for node, mem in zip(valid_nodes.tolist(), memberships):
        for c in mem:
            cluster_to_nodes[int(c)].add(int(node))
    return [s for s in cluster_to_nodes if len(s) > 0]

def _community_size_stats(pred_comms_ge2: list[set[int]]):
    """Return (median, p90) of community sizes after >=2 filter."""
    if len(pred_comms_ge2) == 0:
        return (np.nan, np.nan)
    sizes = np.array([len(c) for c in pred_comms_ge2], dtype=float)
    return (float(np.median(sizes)), float(np.percentile(sizes, 90)))

def _safe_internal_metrics(Z_valid: np.ndarray, labels: np.ndarray):
    try:
        if Z_valid.shape[0] < 2:
            return (np.nan, np.nan, np.nan)
        uniq = np.unique(labels)
        if uniq.size < 2:
            return (np.nan, np.nan, np.nan)
        sil = float(silhouette_score(Z_valid, labels))
        ch  = float(calinski_harabasz_score(Z_valid, labels))
        db  = float(davies_bouldin_score(Z_valid, labels))
        return (sil, ch, db)
    except Exception:
        return (np.nan, np.nan, np.nan)

def _get_task_id() -> int:
    """
    Priority:
      1) --index N
      2) SLURM_ARRAY_TASK_ID
    """
    if "--index" in sys.argv:
        i = sys.argv.index("--index")
        return int(sys.argv[i + 1])

    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is None:
        raise RuntimeError("No SLURM_ARRAY_TASK_ID. Run with --index N or as a SLURM array job.")
    return int(env_idx)

def _decode_task(tid: int):
    """
    tid in [0..31] -> (lambda_key, multiplier, seed)
    4 lambdas × 4 multipliers × 2 seeds = 32 tasks
    """
    if not (0 <= tid < 32):
        raise IndexError(f"Task id {tid} out of range. Expected 0..31.")

    lam_i = tid // 8
    rem = tid % 8
    mult_i = rem // 2
    seed_i = rem % 2

    lambda_key = LAMBDA_KEYS[lam_i]
    multiplier = MULTIPLIERS[mult_i]
    seed = SEEDS[seed_i]
    return lambda_key, float(multiplier), int(seed)


# ------------------------
# Main
# ------------------------
def main():
    out_dir = _ensure_dirs()
    tid = _get_task_id()
    lambda_key, multiplier, seed = _decode_task(tid)

    # Build cfg for this one run (one lambda varied, others fixed)
    cfg = {k: float(FIXED_CFG[k]) for k in LAMBDA_KEYS}
    base_val = float(FIXED_CFG[lambda_key])
    cfg[lambda_key] = multiplier * base_val  # includes 0 = ablation

    # Unique per-task output (no collisions)
    mult_tag = str(multiplier).replace(".", "p")
    csv_path = os.path.join(out_dir, f"run_{lambda_key}_mult{mult_tag}_seed{seed}.csv")
    error_path = os.path.join(out_dir, f"run_{lambda_key}_mult{mult_tag}_seed{seed}_error.txt")

    print(
        f"[RUN] tid={tid} | {lambda_key} x{multiplier} (base {base_val}) -> {cfg[lambda_key]} | "
        f"seed={seed} | EPOCHS={EPOCHS} LR={LR} | eval K={K_FIXED} r={R_FIXED} | out={csv_path}"
    )
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        set_seed(seed)

        # --- Load data ---
        x, edge_index, y, edge_weight, G, communities, y_multi, mask, nid2idx, idx2nid = load_ppi_cp_graph(
            use_onehot=True, device=device
        )

        # valid nodes = masked/labeled nodes (same logic as your KSweep runner)
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask, dtype=bool)

        valid_nodes = np.nonzero(mask_np)[0].astype(int)
        valid_nodes_set = set(valid_nodes.tolist())

        # GT communities restricted to labeled nodes, singleton filtered (>=2)
        gt_ge2 = []
        for c in communities:
            cc = set(c) & valid_nodes_set
            if len(cc) >= 2:
                gt_ge2.append(cc)

        # --- Train once ---
        model = CAGATAblationModel(
            in_dim=x.size(1),
            hidden_dim=8,
            out_dim=K_FIXED,   # keep consistent with your PPI runners
            heads=4
        ).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=LR)

        model.train()
        last_logs = None
        for _ in range(EPOCHS):
            optim.zero_grad()
            _, _, embeddings = model(x, edge_index, edge_weight)
            loss, logs = combined_community_loss_PPI(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_weight=edge_weight,
                lambda_mod=cfg["lambda_mod"],
                lambda_lap=cfg["lambda_lap"],
                lambda_contrast=cfg["lambda_contrast"],
                lambda_orth=cfg["lambda_orth"],
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

        Z = to_numpy(embeddings)
        Z_valid = Z[valid_nodes]

        # --- Fixed-K KMeans on valid nodes ---
        km = KMeans(n_clusters=int(K_FIXED), random_state=seed, n_init=10)
        km.fit(Z_valid)
        labels = km.labels_
        centroids = km.cluster_centers_

        # Internal metrics
        sil, ch, db = _safe_internal_metrics(Z_valid, labels)

        # --- top-r memberships (r fixed = 2) ---
        H = Z_valid
        C = centroids
        H2 = (H * H).sum(axis=1, keepdims=True)
        C2 = (C * C).sum(axis=1, keepdims=True).T
        D2 = np.maximum(H2 + C2 - 2.0 * (H @ C.T), 0.0)
        order = np.argsort(D2, axis=1)
        topr = order[:, :R_FIXED]

        memberships = [row.tolist() for row in topr]
        pred_comms = _build_pred_comms_from_memberships(valid_nodes, memberships, K=int(K_FIXED))
        pred_ge2 = _filter_comms_ge2(pred_comms)

        # Community size stats (GE2)
        med, p90 = _community_size_stats(pred_ge2)

        # ONMI (MGH) on GE2 communities
        if len(valid_nodes) < 2 or len(gt_ge2) == 0 or len(pred_ge2) == 0:
            onmi = np.nan
            onmi_reason = "degenerate_mask_or_empty_gt_or_pred"
        else:
            onmi = float(onmi_mgh(gt_ge2, pred_ge2, nodes_mask=valid_nodes))
            onmi_reason = "" if not np.isnan(onmi) else "nan_onmi"

        # training diagnostics (optional, lightweight)
        extra_logs = {}
        if last_logs is not None:
            extra_logs = {
                "TrainLoss_Final": round(last_logs.get("loss_total", float("nan")), 6),
                "Contrast_PctValid": round(100.0 * last_logs.get("contrast_pct_valid", 0.0), 2),
                "Contrast_PosPerRow": round(last_logs.get("contrast_mean_pos_per_row", 0.0), 4),
            }

        row = {
            "TaskID": int(tid),
            "Seed": int(seed),
            "LambdaKey": lambda_key,
            "Multiplier": float(multiplier),
            "K": int(K_FIXED),
            "r": int(R_FIXED),

            "lambda_mod": cfg["lambda_mod"],
            "lambda_lap": cfg["lambda_lap"],
            "lambda_contrast": cfg["lambda_contrast"],
            "lambda_orth": cfg["lambda_orth"],

            "ONMI_r2": (None if np.isnan(onmi) else round(float(onmi), 6)),
            "ONMI_reason": onmi_reason,

            "Silhouette": (None if np.isnan(sil) else round(float(sil), 6)),
            "CH": (None if np.isnan(ch) else round(float(ch), 2)),
            "DB": (None if np.isnan(db) else round(float(db), 4)),

            "PredComms_GE2": int(len(pred_ge2)),
            "PredSize_Median_GE2": (None if np.isnan(med) else round(float(med), 2)),
            "PredSize_P90_GE2": (None if np.isnan(p90) else round(float(p90), 2)),

            "ValidNodes": int(len(valid_nodes)),
            "GTComms_GE2": int(len(gt_ge2)),

            **extra_logs,
        }

        pd.DataFrame([row]).to_csv(csv_path, index=False)
        print(f"[OK] Wrote {csv_path}")

    except Exception as e:
        err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        with open(error_path, "w") as f:
            f.write(f"[ERROR] tid={tid} seed={seed} lambda_key={lambda_key} multiplier={multiplier}\n")
            f.write(err_msg)
        print(f"[FAIL] Wrote error log at {error_path}")
        raise


if __name__ == "__main__":
    main()
    os._exit(0)
