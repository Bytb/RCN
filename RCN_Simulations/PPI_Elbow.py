# ---------------- experiments/GAT/Simulations/RCN_Simulations/PPI_Elbow.py ----------------

import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# --- Your project imports ---
from Models.RCN import CAGATAblationModel
from Models.LossFunctions import combined_community_loss
from HelperFunctions import cluster_with_kmeans, onmi_mgh
from Data.PPI import load_ppi_graph  # <- your PPI loader

# ------------------------
# Config
# ------------------------
dataset = "PPI"
SEEDS = [120, 328476, 42]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter Grid (keep identical style to your prior script)
lambda_mods = [0.6]                              # fixed
lambda_laps = [0.001, 0.0001, 1e-5]              # Laplacian/L2 (in-loss)
lambda_contrasts = [0.001, 0.0001, 1e-5]
lambda_orth = [0.001, 0.0001, 1e-5, 1e-6]
sweep_combos = list(product(lambda_mods, lambda_laps, lambda_contrasts, lambda_orth))

# ------------------------
# Utilities
# ------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def choose_optimal_k(embeddings: torch.Tensor, k_min: int = 2, k_max: int = 10):
    """
    Pick k by sweeping k-means on the final embeddings and maximizing Silhouette.
    Tie-breakers: Calinski–Harabasz (higher better), then Davies–Bouldin (lower better).
    """
    Z = to_numpy(embeddings)
    best = {"k": None, "sil": -1.0, "ch": -1.0, "db": np.inf}
    for k in range(k_min, max(k_min, k_max) + 1):
        preds = cluster_with_kmeans(embeddings, k)
        try:
            sil = silhouette_score(Z, preds)
        except Exception:
            sil = -1.0
        try:
            ch = calinski_harabasz_score(Z, preds)
        except Exception:
            ch = -1.0
        try:
            db = davies_bouldin_score(Z, preds)
        except Exception:
            db = np.inf

        better = (
            (sil > best["sil"]) or
            (np.isclose(sil, best["sil"]) and ch > best["ch"]) or
            (np.isclose(sil, best["sil"]) and np.isclose(ch, best["ch"]) and db < best["db"])
        )
        if better:
            best = {"k": k, "sil": sil, "ch": ch, "db": db}
    return best["k"], best

def build_predicted_communities(preds: np.ndarray) -> list[set[int]]:
    """Convert k-means label vector into a list of node-index sets."""
    pred_comms = {}
    for idx, lab in enumerate(preds):
        pred_comms.setdefault(int(lab), set()).add(int(idx))
    # Preserve label order by sorted key
    return [pred_comms[k] for k in sorted(pred_comms.keys())]

def evaluate_at_k(
    embeddings: torch.Tensor,
    communities_true: list[set[int]],
    mask: torch.Tensor,
    k: int,
):
    """
    Compute ONMI (MGH) on masked nodes and Silhouette on all nodes.
    Returns: onmi, silhouette, preds, diagnostics_dict
    """
    # K-means predictions
    preds = cluster_with_kmeans(embeddings, k)
    Z = to_numpy(embeddings)

    # Silhouette on ALL embeddings (unchanged)
    sil = silhouette_score(Z, preds)

    # Prepare masked universe
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.asarray(mask, dtype=bool)
    valid_nodes = np.nonzero(mask_np)[0].tolist()

    # Ground-truth: remove singleton complexes, then intersect with mask; drop empties
    gt_masked = []
    for c in communities_true:
        cc = set(c) & set(valid_nodes)
        if len(cc) >= 2:
            gt_masked.append(cc)

    # Predicted: intersect with mask; drop empties
    pred_comms = build_predicted_communities(to_numpy(preds))
    pred_masked = []
    for c in pred_comms:
        cc = set(c) & set(valid_nodes)
        if len(cc) > 0:
            pred_masked.append(cc)

    # Diagnostics and degenerate checks
    diag = {
        "ONMI_valid_nodes": int(len(valid_nodes)),
        "ONMI_GT_comms": int(len(gt_masked)),
        "ONMI_Pred_comms": int(len(pred_masked)),
        "ONMI_reason": "",
    }

    if len(valid_nodes) < 2:
        return (float("nan"), sil, preds, {**diag, "ONMI_reason": "no_masked_nodes"})
    if len(gt_masked) == 0:
        return (float("nan"), sil, preds, {**diag, "ONMI_reason": "empty_gt"})
    if len(pred_masked) == 0:
        return (float("nan"), sil, preds, {**diag, "ONMI_reason": "empty_pred"})

    # ONMI (MGH) on masked partitions
    onmi = onmi_mgh(gt_masked, pred_masked, nodes_mask=valid_nodes)
    if np.isnan(onmi):
        diag["ONMI_reason"] = "zero_entropy_or_degenerate"

    return onmi, sil, preds, diag

# ------------------------
# Data (PPI, One-Hot features)
# ------------------------
x, edge_index, y, edge_weight, G, communities, y_multi, mask, nid2idx, idx2nid = \
    load_ppi_graph(use_onehot=True, device=device)

x, edge_index, y, edge_weight = x.to(device), edge_index.to(device), y.to(device), edge_weight.to(device)

# Overcomplete head for training; we will select k* post-hoc from embeddings
HEAD_OUT_DIM = 32
N = G.number_of_nodes()
K_SWEEP_MAX = min(HEAD_OUT_DIM, max(10, int(np.sqrt(N))))  # same heuristic

results = []

# ------------------------
# Sweep
# ------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("../../../OneDrive - University of South Florida/RCN_Neurips/RCN_Simulations/logs", dataset, f"RCN_Sweep_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

pbar = tqdm(sweep_combos, total=len(sweep_combos), desc="λ-grid")
for lm, ll, lc, lo in pbar:
    pbar.set_postfix(mod=lm, lap=ll, con=lc, orth=lo)

    for seed in SEEDS:
        set_seed(seed)

        # Model with RNBRW-biased attention (pre-softmax), overcomplete head
        model = CAGATAblationModel(
            in_dim=x.size(1),
            hidden_dim=8,
            out_dim=HEAD_OUT_DIM,
            heads=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        # Train
        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            _, _, embeddings = model(x, edge_index, edge_weight)
            loss = combined_community_loss(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_weight=edge_weight,
                lambda_mod=lm,
                lambda_lap=ll,
                lambda_contrast=lc,
                lambda_orth=lo,
                contrast_tau=0.5,
                contrast_variant="node",
            )
            loss.backward()
            optimizer.step()

        # Final embeddings
        model.eval()
        with torch.no_grad():
            _, _, embeddings = model(x, edge_index, edge_weight)

        # --------- Optimal-K selection on embeddings (unchanged) ----------
        k_opt, k_stats = choose_optimal_k(embeddings, k_min=2, k_max=K_SWEEP_MAX)

        # Evaluate at k*: ONMI (masked) + Silhouette (all)
        onmi, sil, preds, diag = evaluate_at_k(
            embeddings=embeddings,
            communities_true=communities,
            mask=mask,
            k=k_opt,
        )

        row = {
            "Graph": dataset,
            "Seed": seed,
            "k_opt": k_opt,
            "λ_mod": lm,
            "λ_lap": ll,
            "λ_contrast": lc,
            "λ_orth": lo,
            "ONMI": (None if np.isnan(onmi) else round(float(onmi), 4)),
            "Silhouette": round(float(sil), 4),
            "Sil_at_k_opt": round(float(k_stats["sil"]), 4),
            "CH_at_k_opt": (round(float(k_stats["ch"]), 4) if np.isfinite(k_stats["ch"]) else k_stats["ch"]),
            "DB_at_k_opt": (round(float(k_stats["db"]), 4) if np.isfinite(k_stats["db"]) else k_stats["db"]),
            # Diagnostics
            "ONMI_valid_nodes": diag["ONMI_valid_nodes"],
            "ONMI_GT_comms": diag["ONMI_GT_comms"],
            "ONMI_Pred_comms": diag["ONMI_Pred_comms"],
            "ONMI_reason": diag["ONMI_reason"],
        }
        results.append(row)

# ------------------------
# Save Results
# ------------------------
df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = os.path.join(out_dir, "results.csv")
df.to_csv(out_csv, index=False)
print(f"✅ Results saved to {out_csv}")
