# RCN_Simulations/OverlapPoC_ArrayRunner.py
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
K_FIXED = 793
EPOCHS = 200
LR = 0.005

CONTRAST_TAU = 0.5
CONTRAST_VARIANT = "node"

# Option 1: Top-r nearest centroids (extend to r=1..10)
TOPR_LIST = list(range(1, 11))

# Option 2a: absolute threshold on softmax weights
# Much smaller tau, and much larger T
TAU_LIST_ABS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
T_LIST_ABS = [1.0, 10.0, 20.0, 50.0, 100.0]

# Option 2b: relative threshold (scale-free)
# Keep k if w_ik >= alpha * max(w_i). Use T=1 for this mode unless you later ask otherwise.
ALPHA_LIST_REL = [0.5, 0.3, 0.2, 0.1]
T_REL = 1.0

MIN_MEMBERSHIPS = 1  # always keep at least one cluster (argmax fallback)


# ------------------------
# PoC hard-coded hyperparams + seeds (3 configs x 2 seeds = 6 array tasks)
# ------------------------
POC_CONFIGS = [
    {"lambda_mod": 0.05, "lambda_lap": 1e-5, "lambda_contrast": 0.1,  "lambda_orth": 1e-6},
    {"lambda_mod": 0.10, "lambda_lap": 1e-5, "lambda_contrast": 0.05, "lambda_orth": 1e-6},
    {"lambda_mod": 0.10, "lambda_lap": 1e-6, "lambda_contrast": 0.05, "lambda_orth": 1e-6},
]
POC_SEEDS = [120, 42]


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

def _slurm_index_or_arg():
    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is not None:
        return int(env_idx)
    if "--index" in sys.argv:
        i = sys.argv.index("--index")
        return int(sys.argv[i + 1])
    raise RuntimeError("No SLURM_ARRAY_TASK_ID; run with --index N for local testing")

def _ensure_dirs():
    out_dir = os.path.join(REPO_DIR, "RCN_Simulations", "logs", OUT_FOLDER_NAME, "partial")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _idx_to_config_seed(idx: int):
    n_seeds = len(POC_SEEDS)
    n_cfgs = len(POC_CONFIGS)
    total = n_cfgs * n_seeds
    if idx < 0 or idx >= total:
        raise IndexError(f"Array index {idx} out of range [0, {total-1}] "
                         f"(configs={n_cfgs}, seeds={n_seeds})")
    cfg_i = idx // n_seeds
    seed_i = idx % n_seeds
    return cfg_i, POC_CONFIGS[cfg_i], POC_SEEDS[seed_i], total

def _pairwise_sq_dists(H: np.ndarray, C: np.ndarray) -> np.ndarray:
    H2 = (H * H).sum(axis=1, keepdims=True)          # [n, 1]
    C2 = (C * C).sum(axis=1, keepdims=True).T        # [1, K]
    D2 = H2 + C2 - 2.0 * (H @ C.T)                   # [n, K]
    return np.maximum(D2, 0.0)

def _softmax_rows(X: np.ndarray) -> np.ndarray:
    X = X - X.max(axis=1, keepdims=True)
    E = np.exp(X)
    return E / (E.sum(axis=1, keepdims=True) + 1e-12)

def _build_pred_comms_from_memberships(valid_nodes: np.ndarray, memberships: list[list[int]], K: int):
    cluster_to_nodes = [set() for _ in range(K)]
    for node, mem in zip(valid_nodes.tolist(), memberships):
        for c in mem:
            cluster_to_nodes[int(c)].add(int(node))
    return [s for s in cluster_to_nodes if len(s) > 0]

def _avg_memberships(memberships: list[list[int]]) -> float:
    if len(memberships) == 0:
        return float("nan")
    return float(np.mean([len(m) for m in memberships]))

def _filter_comms_ge2(comms: list[set[int]]) -> list[set[int]]:
    return [set(c) for c in comms if len(c) >= 2]

def _size_stats_after_filter(pred_comms_ge2: list[set[int]]):
    if len(pred_comms_ge2) == 0:
        return (np.nan, np.nan)
    sizes = np.array([len(c) for c in pred_comms_ge2], dtype=float)
    return (float(np.median(sizes)), float(np.percentile(sizes, 90)))


# ------------------------
# Main
# ------------------------
def main():
    out_dir = _ensure_dirs()
    idx = _slurm_index_or_arg()
    cfg_i, cfg, seed, total_jobs = _idx_to_config_seed(idx)

    lm = float(cfg["lambda_mod"])
    ll = float(cfg["lambda_lap"])
    lc = float(cfg["lambda_contrast"])
    lo = float(cfg["lambda_orth"])

    csv_path   = os.path.join(out_dir, f"run_{idx:04d}.csv")
    error_path = os.path.join(out_dir, f"run_{idx:04d}_error.txt")

    # Skip if already done
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if len(df) >= 1:
                print(f"[SKIP] index={idx} already completed -> {csv_path}")
                return
        except Exception:
            pass

    print(f"[RUN] index={idx}/{total_jobs-1} cfg={cfg_i} seed={seed} "
          f"λ_mod={lm} λ_lap={ll} λ_contrast={lc} λ_orth={lo}")
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

        # GT communities for ONMI: restrict to labeled nodes, then filter singleton communities (>=2)
        gt_ge2 = []
        for c in communities:
            cc = set(c) & valid_nodes_set
            if len(cc) >= 2:
                gt_ge2.append(cc)
        gt_dedup_ge2 = set(frozenset(c) for c in gt_ge2)

        # --- Model ---
        model = CAGATAblationModel(
            in_dim=x.size(1),
            hidden_dim=8,
            out_dim=K_FIXED,
            heads=4
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)

        # --- Train ---
        model.train()
        last_logs = None
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
            last_logs = logs
            loss.backward()
            optim.step()

        # --- Final embeddings ---
        model.eval()
        with torch.no_grad():
            _, _, embeddings = model(x, edge_index, edge_weight)
        Z = to_numpy(embeddings)  # [N, d]

        # --- KMeans base partition (K=793) ---
        km = KMeans(n_clusters=K_FIXED, random_state=seed, n_init=10)
        preds = km.fit_predict(Z)
        centroids = km.cluster_centers_

        # Partition metrics on all nodes
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

        # Distances for valid nodes only
        Z_valid = Z[valid_nodes]
        D2 = _pairwise_sq_dists(Z_valid, centroids)  # [n_valid, K]
        order = np.argsort(D2, axis=1)

        # -------------------------
        # Option 1: top-r
        # -------------------------
        onmi_topr = {}
        avgmem_topr = {}
        pred_ge2_cnt_topr = {}
        med_topr = {}
        p90_topr = {}

        for r in TOPR_LIST:
            topr = order[:, :r]
            memberships = [row.tolist() for row in topr]
            pred_comms = _build_pred_comms_from_memberships(valid_nodes, memberships, K_FIXED)

            pred_ge2 = _filter_comms_ge2(pred_comms)
            pred_ge2_cnt_topr[r] = int(len(pred_ge2))
            avgmem_topr[r] = _avg_memberships(memberships)

            med, p90 = _size_stats_after_filter(pred_ge2)
            med_topr[r], p90_topr[r] = med, p90

            if len(valid_nodes) < 2 or len(gt_ge2) == 0 or len(pred_ge2) == 0:
                onmi_topr[r] = np.nan
            else:
                onmi_topr[r] = float(onmi_mgh(gt_ge2, pred_ge2, nodes_mask=valid_nodes))

        # -------------------------
        # Option 2a: absolute tau threshold over multiple T
        # -------------------------
        onmi_abs = {}          # keyed by (T, tau)
        avgmem_abs = {}
        pred_ge2_cnt_abs = {}
        med_abs = {}
        p90_abs = {}

        for T in T_LIST_ABS:
            if T <= 0:
                raise ValueError("Temperature T must be > 0")
            W = _softmax_rows(-D2 / float(T))  # [n_valid, K]

            for tau in TAU_LIST_ABS:
                memberships = []
                for i in range(W.shape[0]):
                    idxs = np.where(W[i] > float(tau))[0].tolist()
                    if len(idxs) < MIN_MEMBERSHIPS:
                        idxs = [int(np.argmax(W[i]))]
                    memberships.append(idxs)

                pred_comms = _build_pred_comms_from_memberships(valid_nodes, memberships, K_FIXED)
                pred_ge2 = _filter_comms_ge2(pred_comms)

                key = (float(T), float(tau))
                pred_ge2_cnt_abs[key] = int(len(pred_ge2))
                avgmem_abs[key] = _avg_memberships(memberships)
                med, p90 = _size_stats_after_filter(pred_ge2)
                med_abs[key], p90_abs[key] = med, p90

                if len(valid_nodes) < 2 or len(gt_ge2) == 0 or len(pred_ge2) == 0:
                    onmi_abs[key] = np.nan
                else:
                    onmi_abs[key] = float(onmi_mgh(gt_ge2, pred_ge2, nodes_mask=valid_nodes))

        # -------------------------
        # Option 2b: relative alpha threshold at T_REL
        # -------------------------
        onmi_rel = {}          # keyed by alpha
        avgmem_rel = {}
        pred_ge2_cnt_rel = {}
        med_rel = {}
        p90_rel = {}

        if T_REL <= 0:
            raise ValueError("T_REL must be > 0")
        Wrel = _softmax_rows(-D2 / float(T_REL))

        for alpha in ALPHA_LIST_REL:
            memberships = []
            for i in range(Wrel.shape[0]):
                row = Wrel[i]
                mx = float(np.max(row))
                thr = float(alpha) * mx
                idxs = np.where(row >= thr)[0].tolist()
                if len(idxs) < MIN_MEMBERSHIPS:
                    idxs = [int(np.argmax(row))]
                memberships.append(idxs)

            pred_comms = _build_pred_comms_from_memberships(valid_nodes, memberships, K_FIXED)
            pred_ge2 = _filter_comms_ge2(pred_comms)

            pred_ge2_cnt_rel[alpha] = int(len(pred_ge2))
            avgmem_rel[alpha] = _avg_memberships(memberships)
            med, p90 = _size_stats_after_filter(pred_ge2)
            med_rel[alpha], p90_rel[alpha] = med, p90

            if len(valid_nodes) < 2 or len(gt_ge2) == 0 or len(pred_ge2) == 0:
                onmi_rel[alpha] = np.nan
            else:
                onmi_rel[alpha] = float(onmi_mgh(gt_ge2, pred_ge2, nodes_mask=valid_nodes))

        # -------------------------
        # Output row
        # -------------------------
        row = {
            "Graph": DATASET,
            "ConfigID": int(cfg_i),
            "Seed": int(seed),
            "K": int(K_FIXED),
            "λ_mod": lm,
            "λ_lap": ll,
            "λ_contrast": lc,
            "λ_orth": lo,
            "Epochs": int(EPOCHS),
            "LR": float(LR),

            "ValidNodes": int(len(valid_nodes)),
            "GT_Comms_GE2": int(len(gt_ge2)),
            "GT_Dedup_Comms_GE2": int(len(gt_dedup_ge2)),

            "Silhouette": (None if np.isnan(sil) else round(sil, 4)),
            "CalinskiHarabasz": (None if np.isnan(ch) else round(ch, 4)),
            "DaviesBouldin": (None if np.isnan(db) else round(db, 4)),

            "DistType": "sq_euclidean",
        }

        # Option 1 columns
        for r in TOPR_LIST:
            v = onmi_topr.get(r, np.nan)
            m = avgmem_topr.get(r, np.nan)
            row[f"ONMI_topr_r{r}"] = (None if np.isnan(v) else round(float(v), 6))
            row[f"AvgMem_topr_r{r}"] = (None if np.isnan(m) else round(float(m), 4))
            row[f"PredComms_GE2_topr_r{r}"] = int(pred_ge2_cnt_topr.get(r, 0))
            med = med_topr.get(r, np.nan)
            p90 = p90_topr.get(r, np.nan)
            row[f"PredSizeMed_GE2_topr_r{r}"] = (None if np.isnan(med) else round(float(med), 2))
            row[f"PredSizeP90_GE2_topr_r{r}"] = (None if np.isnan(p90) else round(float(p90), 2))

        # Option 2a columns (abs tau + T sweep)
        for T in T_LIST_ABS:
            for tau in TAU_LIST_ABS:
                key = (float(T), float(tau))
                v = onmi_abs.get(key, np.nan)
                m = avgmem_abs.get(key, np.nan)

                tkey = f"T{int(T)}" if float(T).is_integer() else f"T{str(T).replace('.','p')}"
                taukey = f"{tau:.0e}".replace("+", "")  # 1e-04 -> 1e-04
                colbase = f"abs_{tkey}_tau{taukey}"

                row[f"ONMI_{colbase}"] = (None if np.isnan(v) else round(float(v), 6))
                row[f"AvgMem_{colbase}"] = (None if np.isnan(m) else round(float(m), 4))
                row[f"PredComms_GE2_{colbase}"] = int(pred_ge2_cnt_abs.get(key, 0))

                med = med_abs.get(key, np.nan)
                p90 = p90_abs.get(key, np.nan)
                row[f"PredSizeMed_GE2_{colbase}"] = (None if np.isnan(med) else round(float(med), 2))
                row[f"PredSizeP90_GE2_{colbase}"] = (None if np.isnan(p90) else round(float(p90), 2))

        # Option 2b columns (relative alpha at T_REL)
        for alpha in ALPHA_LIST_REL:
            v = onmi_rel.get(alpha, np.nan)
            m = avgmem_rel.get(alpha, np.nan)
            akey = str(alpha).replace(".", "p")
            colbase = f"rel_T{int(T_REL)}_a{akey}"

            row[f"ONMI_{colbase}"] = (None if np.isnan(v) else round(float(v), 6))
            row[f"AvgMem_{colbase}"] = (None if np.isnan(m) else round(float(m), 4))
            row[f"PredComms_GE2_{colbase}"] = int(pred_ge2_cnt_rel.get(alpha, 0))

            med = med_rel.get(alpha, np.nan)
            p90 = p90_rel.get(alpha, np.nan)
            row[f"PredSizeMed_GE2_{colbase}"] = (None if np.isnan(med) else round(float(med), 2))
            row[f"PredSizeP90_GE2_{colbase}"] = (None if np.isnan(p90) else round(float(p90), 2))

        # training debug logs
        if last_logs is not None:
            row.update({
                "TrainLoss_Final": round(last_logs.get("loss_total", float("nan")), 6),
                "Loss_mod": round(last_logs.get("loss_mod", float("nan")), 6),
                "Loss_lap": round(last_logs.get("loss_lap", float("nan")), 6),
                "Loss_contrast": round(last_logs.get("loss_contrast", float("nan")), 6),
                "Loss_orth": round(last_logs.get("loss_orth", float("nan")), 6),
            })

        pd.DataFrame([row]).to_csv(csv_path, index=False)
        print(f"[OK] Wrote {csv_path}")

    except Exception as e:
        err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        with open(error_path, "w") as f:
            f.write(f"[ERROR] index={idx}\n")
            f.write(err_msg)

        nan_row = {
            "Graph": DATASET,
            "ConfigID": None,
            "Seed": None,
            "K": int(K_FIXED),
            "ONMI_reason": f"EXCEPTION: {type(e).__name__}",
        }
        pd.DataFrame([nan_row]).to_csv(csv_path, index=False)
        print(f"[FAIL] index={idx} -> wrote NaN CSV and error log at {error_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    os._exit(0)
