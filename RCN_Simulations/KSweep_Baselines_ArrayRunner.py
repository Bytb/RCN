# ==============================================
# 2) NEW SCRIPT: RCN_Simulations/KSweep_Baselines_ArrayRunner.py
# ==============================================
import os, sys, traceback
import random
import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

# --- project imports ---
from Models.RCN import CAGATAblationModel
from Models.LossFunctions import combined_community_loss_PPI
from Models.GAT_Attention import GAT
from Models.BaselineModels import DMoNModel
from HelperFunctions import onmi_mgh
from Data.PPI import load_ppi_cp_graph


# ------------------------
# Constants / Paths
# ------------------------
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET = "PPI_CP"
OUT_FOLDER_NAME = "PPI_CP_OverlapPoC_Baselines"

EPOCHS = 200
LR = 0.005

# RCN loss config (professor instruction)
CONTRAST_TAU = 0.5
CONTRAST_VARIANT = "node"
FIXED_CFG = {"lambda_mod": 0.05, "lambda_lap": 1e-5, "lambda_contrast": 0.1, "lambda_orth": 1e-6}

# Seeds (2-task run intended)
SEEDS = [120, 42]

# K sweep
K_SWEEP = [100, 200, 300, 400, 600, 793]

# r sweep
R_LIST = [1, 2, 3, 4, 5]

# If this script is placed into a larger array, allow offset mapping like your other runner
KSWEEP_DEFAULT_OFFSET = 6


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
    return [set(c) for c in comms if len(c) >= 2]

def _build_pred_comms_from_memberships(valid_nodes: np.ndarray, memberships: list[list[int]], K: int):
    cluster_to_nodes = [set() for _ in range(K)]
    for node, mem in zip(valid_nodes.tolist(), memberships):
        for c in mem:
            cluster_to_nodes[int(c)].add(int(node))
    return [s for s in cluster_to_nodes if len(s) > 0]

def _community_size_stats(pred_comms_ge2: list[set[int]]):
    if len(pred_comms_ge2) == 0:
        return (np.nan, np.nan)
    sizes = np.array([len(c) for c in pred_comms_ge2], dtype=float)
    return (float(np.median(sizes)), float(np.percentile(sizes, 90)))

def _get_seed_index() -> int:
    """
    Seed-index resolution:
      1) --seed-index N
      2) --index N
      3) SLURM_ARRAY_TASK_ID
    with offset handling like your existing script.
    """
    if "--seed-index" in sys.argv:
        i = sys.argv.index("--seed-index")
        return int(sys.argv[i + 1])
    if "--index" in sys.argv:
        i = sys.argv.index("--index")
        return int(sys.argv[i + 1])

    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is None:
        raise RuntimeError("No SLURM_ARRAY_TASK_ID; run with --seed-index N (or --index N).")
    raw = int(env_idx)

    if 0 <= raw < len(SEEDS):
        return raw

    if 0 <= (raw - KSWEEP_DEFAULT_OFFSET) < len(SEEDS):
        return raw - KSWEEP_DEFAULT_OFFSET

    off = os.environ.get("KSWEEP_INDEX_OFFSET")
    if off is not None:
        off_i = int(off)
        if 0 <= (raw - off_i) < len(SEEDS):
            return raw - off_i

    raise IndexError(
        f"Index {raw} out of range. Expected seed-index in [0,{len(SEEDS)-1}] "
        f"or mappable via offset (default {KSWEEP_DEFAULT_OFFSET})."
    )

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

def _model_from_args() -> str:
    """
    Choose baseline model with --model {GAT, DMoN, RCN_W1}
    Default: RCN_W1 (safe)
    """
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        return str(sys.argv[i + 1]).strip()
    return "RCN_W1"

def _csv_name(model_name: str, seed: int) -> str:
    return f"k_sweep_{model_name}_seed{seed}.csv"


# ------------------------
# Baseline runners
# ------------------------
def _prepare_common_data(device):
    # Load data
    x, edge_index, y, edge_weight, G, communities, y_multi, mask, nid2idx, idx2nid = load_ppi_cp_graph(
        use_onehot=True, device=device
    )

    # valid nodes = labeled nodes
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.asarray(mask, dtype=bool)

    valid_nodes = np.nonzero(mask_np)[0].astype(int)
    valid_nodes_set = set(valid_nodes.tolist())

    # GT communities restricted to labeled nodes and filtered >=2
    gt_ge2 = []
    for c in communities:
        cc = set(c) & valid_nodes_set
        if len(cc) >= 2:
            gt_ge2.append(cc)

    return x, edge_index, edge_weight, valid_nodes, valid_nodes_set, gt_ge2

def _eval_overlap_from_order(valid_nodes, gt_ge2, order, K: int):
    """
    Given 'order' = [n_valid, K] argsort indices (closest-first or highest-prob-first),
    compute ONMI_r1, Best_ONMI_r1to5, etc. using the same overlap wrapper.
    """
    onmi_by_r = {}
    pred_ge2_cnt_by_r = {}
    med_by_r = {}
    p90_by_r = {}

    for r in R_LIST:
        topr = order[:, :r]
        memberships = [row.tolist() for row in topr]
        pred_comms = _build_pred_comms_from_memberships(valid_nodes, memberships, K=int(K))

        pred_ge2 = _filter_comms_ge2(pred_comms)
        pred_ge2_cnt_by_r[r] = int(len(pred_ge2))

        med, p90 = _community_size_stats(pred_ge2)
        med_by_r[r], p90_by_r[r] = med, p90

        if len(valid_nodes) < 2 or len(gt_ge2) == 0 or len(pred_ge2) == 0:
            onmi_by_r[r] = np.nan
        else:
            onmi_by_r[r] = float(onmi_mgh(gt_ge2, pred_ge2, nodes_mask=valid_nodes))

    onmi_r1 = onmi_by_r.get(1, np.nan)

    best_r = None
    best_onmi = -np.inf
    for r in R_LIST:
        v = onmi_by_r.get(r, np.nan)
        if np.isnan(v):
            continue
        if v > best_onmi:
            best_onmi = v
            best_r = r

    if best_r is None:
        best_onmi = np.nan
        avgmem_best = np.nan
        pred_ge2_best = 0
        med_best = np.nan
        p90_best = np.nan
        best_r_out = None
    else:
        avgmem_best = float(best_r)
        pred_ge2_best = int(pred_ge2_cnt_by_r[best_r])
        med_best = float(med_by_r[best_r])
        p90_best = float(p90_by_r[best_r])
        best_r_out = int(best_r)

    return onmi_r1, best_onmi, best_r_out, avgmem_best, pred_ge2_best, med_best, p90_best


def run_rcn_w1(seed: int, device):
    """
    RCN baseline with RNBRW weights = 1 (ablation).
    Train once, then KMeans sweep on fixed hidden embeddings h2.
    """
    x, edge_index, edge_weight, valid_nodes, _, gt_ge2 = _prepare_common_data(device)

    # overwrite edge weights with ones
    edge_weight_ones = torch.ones_like(edge_weight)

    lm = float(FIXED_CFG["lambda_mod"])
    ll = float(FIXED_CFG["lambda_lap"])
    lc = float(FIXED_CFG["lambda_contrast"])
    lo = float(FIXED_CFG["lambda_orth"])

    model = CAGATAblationModel(in_dim=x.size(1), hidden_dim=8, out_dim=793, heads=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(EPOCHS):
        optim.zero_grad()
        _, _, embeddings = model(x, edge_index, edge_weight_ones)
        loss, _ = combined_community_loss_PPI(
            embeddings=embeddings,
            edge_index=edge_index,
            edge_weight=edge_weight_ones,
            lambda_mod=lm,
            lambda_lap=ll,
            lambda_contrast=lc,
            lambda_orth=lo,
            contrast_tau=CONTRAST_TAU,
            contrast_variant=CONTRAST_VARIANT,
            return_stats=True
        )
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        _, _, embeddings = model(x, edge_index, edge_weight_ones)

    Z = to_numpy(embeddings)
    Z_valid = Z[valid_nodes]

    rows = []
    for K in K_SWEEP:
        km = KMeans(n_clusters=int(K), random_state=seed, n_init=10)
        km.fit(Z_valid)
        labels = km.labels_
        centroids = km.cluster_centers_

        sil, ch, db = _safe_internal_metrics(Z_valid, labels)

        # distances for top-r order
        H = Z_valid
        C = centroids
        H2 = (H * H).sum(axis=1, keepdims=True)
        C2 = (C * C).sum(axis=1, keepdims=True).T
        D2 = np.maximum(H2 + C2 - 2.0 * (H @ C.T), 0.0)
        order = np.argsort(D2, axis=1)

        onmi_r1, best_onmi, best_r, avgmem_best, pred_ge2_best, med_best, p90_best = _eval_overlap_from_order(
            valid_nodes=valid_nodes,
            gt_ge2=gt_ge2,
            order=order,
            K=int(K)
        )

        rows.append({
            "Seed": int(seed),
            "K": int(K),
            "Silhouette": (None if np.isnan(sil) else round(float(sil), 6)),
            "CH": (None if np.isnan(ch) else round(float(ch), 2)),
            "DB": (None if np.isnan(db) else round(float(db), 4)),
            "ONMI_r1": (None if np.isnan(onmi_r1) else round(float(onmi_r1), 6)),
            "Best_ONMI_r1to5": (None if np.isnan(best_onmi) else round(float(best_onmi), 6)),
            "Best_r": (None if best_r is None else int(best_r)),
            "AvgMem_at_Best": (None if np.isnan(avgmem_best) else round(float(avgmem_best), 4)),
            "PredComms_GE2_at_Best": int(pred_ge2_best),
            "PredSize_Median_GE2_at_Best": (None if np.isnan(med_best) else round(float(med_best), 2)),
            "PredSize_P90_GE2_at_Best": (None if np.isnan(p90_best) else round(float(p90_best), 2)),
        })

    return rows


def run_gat(seed: int, device):
    """
    GAT baseline (Option C):
      - train once
      - extract fixed-dimensional conv1 hidden embedding h1
      - KMeans sweep on h1, same overlap wrapper
    """
    x, edge_index, edge_weight, valid_nodes, _, gt_ge2 = _prepare_common_data(device)

    # out_dim is irrelevant for clustering now (we cluster h1), but must exist
    model = GAT(in_dim=x.size(1), out_dim=793, self_loops=True).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    data = Data(x=x, edge_index=edge_index)

    # Train with the same community loss on the hidden embedding h1 (fixed dim)
    lm = float(FIXED_CFG["lambda_mod"])
    ll = float(FIXED_CFG["lambda_lap"])
    lc = float(FIXED_CFG["lambda_contrast"])
    lo = float(FIXED_CFG["lambda_orth"])

    model.train()
    for _ in range(EPOCHS):
        optim.zero_grad()
        _, _, _, h1 = model(data)  # <-- h1 is the fixed-dimensional embedding
        loss, _ = combined_community_loss_PPI(
            embeddings=h1,                 # use h1 in loss + evaluation
            edge_index=edge_index,
            edge_weight=edge_weight,       # still use RNBRW weights in the loss term
            lambda_mod=lm,
            lambda_lap=ll,
            lambda_contrast=lc,
            lambda_orth=lo,
            contrast_tau=CONTRAST_TAU,
            contrast_variant=CONTRAST_VARIANT,
            return_stats=True
        )
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        _, _, _, h1 = model(data)

    Z = to_numpy(h1)
    Z_valid = Z[valid_nodes]

    rows = []
    for K in K_SWEEP:
        km = KMeans(n_clusters=int(K), random_state=seed, n_init=10)
        km.fit(Z_valid)
        labels = km.labels_
        centroids = km.cluster_centers_

        sil, ch, db = _safe_internal_metrics(Z_valid, labels)

        # distances for top-r order
        H = Z_valid
        C = centroids
        H2 = (H * H).sum(axis=1, keepdims=True)
        C2 = (C * C).sum(axis=1, keepdims=True).T
        D2 = np.maximum(H2 + C2 - 2.0 * (H @ C.T), 0.0)
        order = np.argsort(D2, axis=1)

        onmi_r1, best_onmi, best_r, avgmem_best, pred_ge2_best, med_best, p90_best = _eval_overlap_from_order(
            valid_nodes=valid_nodes,
            gt_ge2=gt_ge2,
            order=order,
            K=int(K)
        )

        rows.append({
            "Seed": int(seed),
            "K": int(K),
            "Silhouette": (None if np.isnan(sil) else round(float(sil), 6)),
            "CH": (None if np.isnan(ch) else round(float(ch), 2)),
            "DB": (None if np.isnan(db) else round(float(db), 4)),
            "ONMI_r1": (None if np.isnan(onmi_r1) else round(float(onmi_r1), 6)),
            "Best_ONMI_r1to5": (None if np.isnan(best_onmi) else round(float(best_onmi), 6)),
            "Best_r": (None if best_r is None else int(best_r)),
            "AvgMem_at_Best": (None if np.isnan(avgmem_best) else round(float(avgmem_best), 4)),
            "PredComms_GE2_at_Best": int(pred_ge2_best),
            "PredSize_Median_GE2_at_Best": (None if np.isnan(med_best) else round(float(med_best), 2)),
            "PredSize_P90_GE2_at_Best": (None if np.isnan(p90_best) else round(float(p90_best), 2)),
        })

    return rows


def run_dmon(seed: int, device):
    """
    DMoN baseline:
      - retrain per K (K is part of the model)
      - use top-r from soft assignments S (no KMeans for overlap memberships)
      - still compute Silhouette/CH/DB on a hard partition derived from argmax(S)
        and distances are based on S for top-r (descending probabilities).
    """
    x, edge_index, edge_weight, valid_nodes, _, gt_ge2 = _prepare_common_data(device)

    # Build dense adjacency for DMoN loss (as you did in your Karate code)
    # NOTE: This may be heavy depending on n_nodes; but matches your baseline usage.
    A = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=x.size(0))[0].to(device)

    rows = []
    for K in K_SWEEP:
        set_seed(seed)

        model = DMoNModel(
            in_dim=x.size(1),
            hidden_dim=64,
            num_clusters=int(K),
            dropout=0.3,
            lambda_orth=0.001
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)

        # Train for this K
        model.train()
        for _ in range(EPOCHS):
            optim.zero_grad()
            S = model(x, edge_index)              # [N, K] soft assignments
            loss = model.dmon_loss(S, A)
            loss.backward()
            optim.step()

        # Final assignments
        model.eval()
        with torch.no_grad():
            S = model(x, edge_index)              # [N, K]

        S_np = to_numpy(S)
        S_valid = S_np[valid_nodes]               # [n_valid, K]

        # Hard labels for internal metrics (argmax)
        labels = np.argmax(S_valid, axis=1)
        sil, ch, db = _safe_internal_metrics(S_valid, labels)

        # For overlap memberships: top-r clusters by probability (descending)
        order = np.argsort(-S_valid, axis=1)      # highest prob first

        onmi_r1, best_onmi, best_r, avgmem_best, pred_ge2_best, med_best, p90_best = _eval_overlap_from_order(
            valid_nodes=valid_nodes,
            gt_ge2=gt_ge2,
            order=order,
            K=int(K)
        )

        rows.append({
            "Seed": int(seed),
            "K": int(K),
            "Silhouette": (None if np.isnan(sil) else round(float(sil), 6)),
            "CH": (None if np.isnan(ch) else round(float(ch), 2)),
            "DB": (None if np.isnan(db) else round(float(db), 4)),
            "ONMI_r1": (None if np.isnan(onmi_r1) else round(float(onmi_r1), 6)),
            "Best_ONMI_r1to5": (None if np.isnan(best_onmi) else round(float(best_onmi), 6)),
            "Best_r": (None if best_r is None else int(best_r)),
            "AvgMem_at_Best": (None if np.isnan(avgmem_best) else round(float(avgmem_best), 4)),
            "PredComms_GE2_at_Best": int(pred_ge2_best),
            "PredSize_Median_GE2_at_Best": (None if np.isnan(med_best) else round(float(med_best), 2)),
            "PredSize_P90_GE2_at_Best": (None if np.isnan(p90_best) else round(float(p90_best), 2)),
        })

    return rows


# ------------------------
# Main
# ------------------------
def main():
    out_dir = _ensure_dirs()
    seed_idx = _get_seed_index()
    seed = SEEDS[seed_idx]

    model_name = _model_from_args().upper()
    csv_path = os.path.join(out_dir, _csv_name(model_name, seed))
    error_path = os.path.join(out_dir, f"{_csv_name(model_name, seed).replace('.csv','')}_error.txt")

    print(f"[RUN] Baseline K-sweep | model={model_name} seed={seed} | "
          f"out={csv_path} | folder={OUT_FOLDER_NAME} | "
          f"EPOCHS={EPOCHS} LR={LR} K_SWEEP={K_SWEEP} R_LIST={R_LIST}")
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        set_seed(seed)

        if model_name == "RCN_W1":
            rows = run_rcn_w1(seed, device)
        elif model_name == "GAT":
            rows = run_gat(seed, device)
        elif model_name == "DMON":
            rows = run_dmon(seed, device)
        else:
            raise ValueError("Unknown --model. Use one of: GAT, DMoN, RCN_W1")

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[OK] Wrote {csv_path}")

    except Exception as e:
        err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        with open(error_path, "w") as f:
            f.write(f"[ERROR] model={model_name} seed={seed}\n")
            f.write(err_msg)
        print(f"[FAIL] model={model_name} seed={seed} -> wrote error log at {error_path}")
        raise


if __name__ == "__main__":
    main()
    os._exit(0)
