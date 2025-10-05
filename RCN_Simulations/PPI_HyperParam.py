import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- project imports ---
from Models.RCN import CAGATAblationModel
from Models.LossFunctions import combined_community_loss
from HelperFunctions import cluster_with_kmeans, onmi_mgh
from Data.PPI import load_ppi_graph  # your PPI loader

# ------------------------
# Config
# ------------------------
dataset = "PPI"
SEED = 42                 # single seed (no averaging)
K_FIXED = 71              # fixed number of clusters for k-means
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter grid
lambda_mods       = [0.6]                         # fixed
lambda_laps       = [0.001, 0.0001, 1e-5]
lambda_contrasts  = [0.001, 0.0001, 1e-5]
lambda_orth       = [0.001, 0.0001, 1e-5, 1e-6]
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

def to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

# ------------------------
# Load PPI
# ------------------------
x, edge_index, y, edge_weight, G, communities, y_multi, mask, nid2idx, idx2nid = \
    load_ppi_graph(use_onehot=True, device=device)

x, edge_index, y, edge_weight = x.to(device), edge_index.to(device), y.to(device), edge_weight.to(device)

# Compute effective GT community count after masking + removing singletons; print it once
if isinstance(mask, torch.Tensor):
    mask_np = mask.detach().cpu().numpy().astype(bool)
else:
    mask_np = np.asarray(mask, dtype=bool)

valid_nodes = np.nonzero(mask_np)[0].tolist()
gt_masked = []
for c in communities:
    cc = set(c) & set(valid_nodes)
    if len(cc) >= 2:
        gt_masked.append(cc)

print(f"[INFO] Effective GT communities after masking (len>=2): {len(gt_masked)}")
print(f"[INFO] Fixed K used for k-means: {K_FIXED}")

# ------------------------
# Sweep
# ------------------------
set_seed(SEED)
results = []

pbar = tqdm(sweep_combos, total=len(sweep_combos), desc="λ-grid")
for lm, ll, lc, lo in pbar:
    pbar.set_postfix(mod=lm, lap=ll, con=lc, orth=lo)

    # Model (no overcomplete head; out_dim = K_FIXED as requested)
    model = CAGATAblationModel(
        in_dim=x.size(1),
        hidden_dim=8,
        out_dim=K_FIXED,
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

    # Fixed-K clustering
    preds = cluster_with_kmeans(embeddings, K_FIXED)
    Z = to_numpy(embeddings)

    # Metrics: Silhouette/CH/DB on all nodes, ONMI (MGH) on masked nodes
    try:
        sil = float(silhouette_score(Z, preds))
    except Exception:
        sil = float("nan")
    try:
        ch = float(calinski_harabasz_score(Z, preds))
    except Exception:
        ch = float("nan")
    try:
        db = float(davies_bouldin_score(Z, preds))
    except Exception:
        db = float("nan")

    # Pred communities for ONMI restricted to masked nodes (drop empties)
    pred_map = {}
    for idx, lab in enumerate(to_numpy(preds)):
        pred_map.setdefault(int(lab), set()).add(int(idx))
    pred_comms = [pred_map[k] for k in sorted(pred_map.keys())]

    pred_masked = []
    for c in pred_comms:
        cc = set(c) & set(valid_nodes)
        if len(cc) > 0:
            pred_masked.append(cc)

    # ONMI diagnostics
    onmi_reason = ""
    if len(valid_nodes) < 2:
        onmi = float("nan"); onmi_reason = "no_masked_nodes"
    elif len(gt_masked) == 0:
        onmi = float("nan"); onmi_reason = "empty_gt"
    elif len(pred_masked) == 0:
        onmi = float("nan"); onmi_reason = "empty_pred"
    else:
        onmi = float(onmi_mgh(gt_masked, pred_masked, nodes_mask=valid_nodes))
        if np.isnan(onmi):
            onmi_reason = "zero_entropy_or_degenerate"

    # Record row
    results.append({
        "Graph": dataset,
        "Seed": SEED,
        "K": K_FIXED,
        "λ_mod": lm,
        "λ_lap": ll,
        "λ_contrast": lc,
        "λ_orth": lo,
        "Silhouette": (None if np.isnan(sil) else round(sil, 4)),
        "CalinskiHarabasz": (None if np.isnan(ch) else round(ch, 4)),
        "DaviesBouldin": (None if np.isnan(db) else round(db, 4)),
        "ONMI": (None if np.isnan(onmi) else round(onmi, 4)),
        "ONMI_valid_nodes": int(len(valid_nodes)),
        "ONMI_GT_comms": int(len(gt_masked)),
        "ONMI_Pred_comms": int(len(pred_masked)),
        "ONMI_reason": onmi_reason,
    })

# ------------------------
# Save Results
# ------------------------
df = pd.DataFrame(results)
out_csv = "PPI_Sweep_results.csv"
df.to_csv(out_csv, index=False)
print(f"✅ Results saved to {out_csv}")
