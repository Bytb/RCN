# RCN_Simulations/KSweep_ArrayRunner.py
import os, sys, traceback
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

DATASET = "PPI_CP"
OUT_FOLDER_NAME = "PPI_CP_OverlapPoC"
EPOCHS = 200
LR = 0.005

CONTRAST_TAU = 0.5
CONTRAST_VARIANT = "node"

# Fixed training config (professor instruction)
FIXED_CFG = {"lambda_mod": 0.05, "lambda_lap": 1e-5, "lambda_contrast": 0.1, "lambda_orth": 1e-6}

# Only two seeds; intended as a 2-task run (seed index 0..1)
SEEDS = [120, 42]

# K sweep only at clustering stage (no retraining)
K_SWEEP = [100, 200, 300, 400, 600, 793]

# r sweep for "best over r=1..5"
R_LIST = [1, 2, 3, 4, 5]

# If you run this KSweep inside a larger array (e.g., PoC is 0..5 and KSweep is 6..7),
# set this offset to 6 (either via env var or it will auto-detect).
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

def _get_seed_index() -> int:
    """
    Robust seed-index resolution.

    Priority:
      1) Explicit --seed-index N (recommended)
      2) Explicit --index N (legacy)
      3) SLURM_ARRAY_TASK_ID
    Then:
      - If idx is in [0..len(SEEDS)-1], use it directly.
      - Else, if idx-6 is in range (common case when KSweep lives at array tasks 6..7),
        use idx-6.
      - Else, if KSWEEP_INDEX_OFFSET env var is set, try idx - OFFSET.
      - Else, raise.
    """
    # 1) --seed-index
    if "--seed-index" in sys.argv:
        i = sys.argv.index("--seed-index")
        return int(sys.argv[i + 1])

    # 2) --index (legacy)
    if "--index" in sys.argv:
        i = sys.argv.index("--index")
        return int(sys.argv[i + 1])

    # 3) SLURM_ARRAY_TASK_ID
    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is None:
        raise RuntimeError("No SLURM_ARRAY_TASK_ID; run with --seed-index N (or --index N).")

    raw = int(env_idx)

    # Direct (0..1)
    if 0 <= raw < len(SEEDS):
        return raw

    # Common combined-array case: PoC uses 0..5 and KSweep uses 6..7
    if 0 <= (raw - KSWEEP_DEFAULT_OFFSET) < len(SEEDS):
        return raw - KSWEEP_DEFAULT_OFFSET

    # Configurable offset via env var
    off = os.environ.get("KSWEEP_INDEX_OFFSET")
    if off is not None:
        off_i = int(off)
        if 0 <= (raw - off_i) < len(SEEDS):
            return raw - off_i

    raise IndexError(
        f"Index {raw} out of range for KSweep. Expected seed-index in [0,{len(SEEDS)-1}] "
        f"or a larger array index that can be mapped via offset (default {KSWEEP_DEFAULT_OFFSET})."
    )


def _safe_internal_metrics(Z_valid: np.ndarray, labels: np.ndarray):
    """
    Compute label-free internal clustering metrics for a hard partition:
      - Silhouette (higher better)
      - CH (higher better)
      - DB (lower better)

    Returns (sil, ch, db) as floats or np.nan if invalid/degenerate.
    """
    try:
        # Need at least 2 clusters present and at least 2 samples
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


# ------------------------
# Main
# ------------------------
def main():
    out_dir = _ensure_dirs()
    seed_idx = _get_seed_index()
    seed = SEEDS[seed_idx]

    lm = float(FIXED_CFG["lambda_mod"])
    ll = float(FIXED_CFG["lambda_lap"])
    lc = float(FIXED_CFG["lambda_contrast"])
    lo = float(FIXED_CFG["lambda_orth"])

    csv_path   = os.path.join(out_dir, f"k_sweep_seed{seed}.csv")
    error_path = os.path.join(out_dir, f"k_sweep_seed{seed}_error.txt")

    print(
        f"[RUN] K-sweep seed={seed} (train once, sweep K={K_SWEEP}, evaluate r={R_LIST}) "
        f"λ_mod={lm} λ_lap={ll} λ_contrast={lc} λ_orth={lo} | "
        f"Embedding provenance: NO checkpoint loaded; embeddings are computed in-memory from the FINAL model "
        f"after EPOCHS={EPOCHS} with LR={LR} and out_dim=793."
    )
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        set_seed(seed)

        # --- Load data ---
        x, edge_index, y, edge_weight, G, communities, y_multi, mask, nid2idx, idx2nid = load_ppi_cp_graph(
            use_onehot=True, device=device
        )

        # valid nodes = all labeled nodes (membership >= 1)
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask, dtype=bool)

        valid_nodes = np.nonzero(mask_np)[0].astype(int)
        valid_nodes_set = set(valid_nodes.tolist())

        # GT communities restricted to labeled nodes, then singleton-community filtered (>=2)
        gt_ge2 = []
        for c in communities:
            cc = set(c) & valid_nodes_set
            if len(cc) >= 2:
                gt_ge2.append(cc)

        # --- Train once (keep model consistent with main PoC runner) ---
        # NOTE: your PoC runner used out_dim=793, so we keep that to stay faithful.
        model = CAGATAblationModel(
            in_dim=x.size(1),
            hidden_dim=8,
            out_dim=793,
            heads=4
        ).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=LR)

        model.train()
        for _ in range(EPOCHS):
            optim.zero_grad()
            _, _, embeddings = model(x, edge_index, edge_weight)
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
            loss.backward()
            optim.step()

        # --- Final embeddings (freeze) ---
        model.eval()
        with torch.no_grad():
            _, _, embeddings = model(x, edge_index, edge_weight)

        Z = to_numpy(embeddings)        # [N, d]
        Z_valid = Z[valid_nodes]        # [n_valid, d]

        # --- Sweep K in KMeans, evaluate top-r overlap for r=1..5 ---
        rows = []
        for K in K_SWEEP:
            km = KMeans(n_clusters=int(K), random_state=seed, n_init=10)
            km.fit(Z_valid)
            labels = km.labels_
            centroids = km.cluster_centers_  # [K, d]

            # Internal label-free metrics (hard partition)
            sil, ch, db = _safe_internal_metrics(Z_valid, labels)

            # Pairwise squared distances to centroids (for top-r)
            H = Z_valid
            C = centroids
            H2 = (H * H).sum(axis=1, keepdims=True)          # [n_valid, 1]
            C2 = (C * C).sum(axis=1, keepdims=True).T        # [1, K]
            D2 = np.maximum(H2 + C2 - 2.0 * (H @ C.T), 0.0)  # [n_valid, K]
            order = np.argsort(D2, axis=1)

            onmi_by_r = {}
            pred_ge2_cnt_by_r = {}
            med_by_r = {}
            p90_by_r = {}

            for r in R_LIST:
                topr = order[:, :r]  # [n_valid, r]
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

            # Best over r=1..5
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
                # For top-r, AvgMem is exactly r (every node gets exactly r memberships)
                avgmem_best = float(best_r)
                pred_ge2_best = int(pred_ge2_cnt_by_r[best_r])
                med_best = float(med_by_r[best_r])
                p90_best = float(p90_by_r[best_r])
                best_r_out = int(best_r)

            rows.append({
                "Seed": int(seed),
                "K": int(K),

                # --- new internal metrics ---
                "Silhouette": (None if np.isnan(sil) else round(float(sil), 6)),
                "CH": (None if np.isnan(ch) else round(float(ch), 2)),
                "DB": (None if np.isnan(db) else round(float(db), 4)),

                # --- existing external/overlap metrics ---
                "ONMI_r1": (None if np.isnan(onmi_r1) else round(float(onmi_r1), 6)),
                "Best_ONMI_r1to5": (None if np.isnan(best_onmi) else round(float(best_onmi), 6)),
                "Best_r": best_r_out,
                "AvgMem_at_Best": (None if np.isnan(avgmem_best) else round(float(avgmem_best), 4)),
                "PredComms_GE2_at_Best": int(pred_ge2_best),
                "PredSize_Median_GE2_at_Best": (None if np.isnan(med_best) else round(float(med_best), 2)),
                "PredSize_P90_GE2_at_Best": (None if np.isnan(p90_best) else round(float(p90_best), 2)),
            })

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[OK] Wrote {csv_path}")

    except Exception as e:
        err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        with open(error_path, "w") as f:
            f.write(f"[ERROR] seed={seed}\n")
            f.write(err_msg)
        print(f"[FAIL] seed={seed} -> wrote error log at {error_path}")
        raise


if __name__ == "__main__":
    main()
    os._exit(0)
