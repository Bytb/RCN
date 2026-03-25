# RCN_Simulations/PPI_CP_RCN_MechanismControls_ArrayRunner.py
"""
RCN mechanism controls for isolating RNBRW/cycle contribution (PPI-CP).

GOAL (IMPORTANT):
  Keep TRAINING + EVALUATION identical to the original KSweep pipeline at the
  matched config, except for edge-weight variant.

Therefore:
  - Train ONCE per run with out_dim = 793
  - Evaluate with KMeans(K=793) on the final embeddings (Z_valid)
  - Assign overlap memberships using nearest centroids (top-r), with r fixed (default r=2)
  - NO K sweep

Variants:
  1) full                  : use loader edge_weight as-is
  2) uniform               : all original undirected edges weight 1
  3) shuffled              : permute original RNBRW weights across original edges
  4) degree_only           : w_uv = 1/sqrt(deg(u)*deg(v)), normalized by mean
  5a) cyclebreak_shuffle   : compute RNBRW-like weights on degree-rewired index graph,
                             then shuffle-assign to original edges
  5b) cyclebreak_degreepair: compute RNBRW-like weights on degree-rewired index graph,
                             then assign by degree-pair matching (nearest fallback)

Parallelization:
  - One task = one (seed, variant)
  - args:
      * --task-index t  (maps across seeds x variants)
      * OR --seed-index i --variant name
      * OR SLURM_ARRAY_TASK_ID as seed-index, variant defaults to full

Outputs:
  - CSV per task:
      RCN_Simulations/logs/PPI_CP_OverlapPoC_Baselines/partial/mechanism_{variant}_seed{seed}.csv
  - Diagnostics per task:
      .../mechanism_{variant}_seed{seed}_diag.txt
"""

from __future__ import annotations

import os
import sys
import math
import traceback
import random
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import networkx as nx

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
OUT_FOLDER_NAME = "PPI_CP_OverlapPoC_Baselines"

EPOCHS = 200
LR = 0.005

CONTRAST_TAU = 0.5
CONTRAST_VARIANT = "node"
FIXED_CFG = {"lambda_mod": 0.05, "lambda_lap": 1e-5, "lambda_contrast": 0.1, "lambda_orth": 1e-6}

SEEDS = [120, 42]

# fixed evaluation (NO K sweep)
K_FIXED = 793
R_FIXED = 2

VARIANTS = [
    "full",
    "uniform",
    "shuffled",
    "degree_only",
    "cyclebreak_shuffle",
    "cyclebreak_degreepair",
]

SWAP_MULT_DEFAULT = 10
RNBRW_WALKS_DEFAULT = 6
RNBRW_WALKLEN_DEFAULT = 8
RNBRW_STARTS_DEFAULT = 4000


# ------------------------
# Utilities
# ------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def _ensure_dirs() -> str:
    out_dir = os.path.join(REPO_DIR, "RCN_Simulations", "logs", OUT_FOLDER_NAME, "partial")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _safe_internal_metrics(Z_valid: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Silhouette (higher), CH (higher), DB (lower)."""
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


def _filter_comms_ge2(comms: List[set]) -> List[set]:
    return [set(c) for c in comms if len(c) >= 2]


def _build_pred_comms_from_memberships(valid_nodes: np.ndarray,
                                       memberships: List[List[int]],
                                       K: int) -> List[set]:
    cluster_to_nodes = [set() for _ in range(K)]
    for node, mem in zip(valid_nodes.tolist(), memberships):
        for c in mem:
            cluster_to_nodes[int(c)].add(int(node))
    return [s for s in cluster_to_nodes if len(s) > 0]


def _community_size_stats(pred_comms_ge2: List[set]) -> Tuple[float, float]:
    if len(pred_comms_ge2) == 0:
        return (np.nan, np.nan)
    sizes = np.array([len(c) for c in pred_comms_ge2], dtype=float)
    return (float(np.median(sizes)), float(np.percentile(sizes, 90)))


def _get_seed_index() -> int:
    if "--seed-index" in sys.argv:
        i = sys.argv.index("--seed-index")
        return int(sys.argv[i + 1])

    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is None:
        raise RuntimeError("No SLURM_ARRAY_TASK_ID; run with --seed-index N.")
    raw = int(env_idx)

    if 0 <= raw < len(SEEDS):
        return raw
    raise IndexError(f"seed-index out of range: {raw} for SEEDS={SEEDS}")


def _get_variant() -> str:
    if "--variant" in sys.argv:
        i = sys.argv.index("--variant")
        v = str(sys.argv[i + 1]).strip().lower()
        if v not in VARIANTS:
            raise ValueError(f"Unknown variant '{v}'. Choose from {VARIANTS}")
        return v
    return "full"


def _get_task_index() -> Optional[int]:
    if "--task-index" in sys.argv:
        i = sys.argv.index("--task-index")
        return int(sys.argv[i + 1])
    return None


def _decode_task(task_idx: int) -> Tuple[int, str]:
    V = len(VARIANTS)
    S = len(SEEDS)
    if task_idx < 0 or task_idx >= S * V:
        raise IndexError(f"task-index {task_idx} out of range [0, {S*V-1}]")
    seed_idx = task_idx // V
    var_idx = task_idx % V
    return seed_idx, VARIANTS[var_idx]


def _coalesce_args(seed_idx: Optional[int], variant: Optional[str]) -> Tuple[int, str]:
    t = _get_task_index()
    if t is not None:
        return _decode_task(t)

    if seed_idx is None:
        seed_idx = _get_seed_index()
    if variant is None:
        variant = _get_variant()
    return seed_idx, variant


# ------------------------
# Graph + labeling prep
# ------------------------
def _prepare_common(device):
    x, edge_index, y, edge_weight, G_raw, communities, y_multi, mask, nid2idx, idx2nid = load_ppi_cp_graph(
        use_onehot=True, device=device
    )

    # valid labeled nodes
    mask_np = mask.detach().cpu().numpy().astype(bool) if isinstance(mask, torch.Tensor) else np.asarray(mask, bool)
    valid_nodes = np.nonzero(mask_np)[0].astype(int)
    valid_nodes_set = set(valid_nodes.tolist())

    # GT comms restricted to labeled nodes, filter >=2
    gt_ge2: List[set] = []
    for c in communities:
        cc = set(c) & valid_nodes_set
        if len(cc) >= 2:
            gt_ge2.append(cc)

    # Build undirected unique edge list u<v; average both directions if present
    ei = edge_index.detach().cpu().numpy()
    ew = edge_weight.detach().cpu().numpy()

    und_pairs: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for k in range(ei.shape[1]):
        u = int(ei[0, k]); v = int(ei[1, k])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        und_pairs[(a, b)].append(float(ew[k]))

    und_edges = np.array(list(und_pairs.keys()), dtype=np.int64)  # [E_und,2]
    und_w = np.array([float(np.mean(und_pairs[tuple(p)])) for p in und_edges], dtype=np.float32)

    num_nodes = int(x.size(0))

    return x, edge_index, edge_weight, num_nodes, und_edges, und_w, valid_nodes, gt_ge2


def _to_directed_weights(edge_index: torch.Tensor,
                         und_edges: np.ndarray,
                         und_w: np.ndarray,
                         device) -> torch.Tensor:
    """Convert undirected weights (u<v list) into directed weights aligned with edge_index columns."""
    m = {(int(u), int(v)): float(w) for (u, v), w in zip(und_edges.tolist(), und_w.tolist())}
    ei = edge_index.detach().cpu().numpy()
    w_dir = np.empty((ei.shape[1],), dtype=np.float32)

    for k in range(ei.shape[1]):
        u = int(ei[0, k]); v = int(ei[1, k])
        if u == v:
            w_dir[k] = 0.0
            continue
        a, b = (u, v) if u < v else (v, u)
        w_dir[k] = m.get((a, b), 1.0)

    return torch.tensor(w_dir, dtype=torch.float32, device=device)


def _build_index_undirected_graph(num_nodes: int, und_edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))
    G.add_edges_from((int(u), int(v)) for u, v in und_edges)
    return G


# ------------------------
# Weight construction per variant
# ------------------------
def _weights_full(und_w: np.ndarray) -> np.ndarray:
    return und_w.copy()


def _weights_uniform(und_w: np.ndarray) -> np.ndarray:
    return np.ones_like(und_w, dtype=np.float32)


def _weights_shuffled(und_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = und_w.copy()
    rng.shuffle(w)
    return w.astype(np.float32)


def _weights_degree_only(und_edges: np.ndarray, num_nodes: int) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=np.int64)
    for u, v in und_edges:
        deg[int(u)] += 1
        deg[int(v)] += 1

    w = np.empty((und_edges.shape[0],), dtype=np.float32)
    for i, (u, v) in enumerate(und_edges):
        du = max(int(deg[int(u)]), 1)
        dv = max(int(deg[int(v)]), 1)
        w[i] = 1.0 / math.sqrt(float(du) * float(dv))

    mu = float(np.mean(w)) if w.size else 1.0
    if mu > 0:
        w = w / mu
    return w.astype(np.float32)


def _rewire_degree_preserving(G_und: nx.Graph, seed: int, swap_mult: int) -> nx.Graph:
    H = G_und.copy()
    E = H.number_of_edges()
    nswap = max(int(swap_mult * E), 1)
    max_tries = max(int(20 * nswap), int(5 * nswap))

    rnd = random.Random(seed)
    try:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=rnd)
    except TypeError:
        random.seed(seed)
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    return H


def _approx_rnbrw_weights(G_und: nx.Graph,
                         seed: int,
                         num_starts: int,
                         num_walks: int,
                         walk_len: int) -> Dict[Tuple[int, int], float]:
    """Approx RNBRW-like weights via non-backtracking walk edge traversal freq."""
    rng = np.random.default_rng(seed)

    nodes = np.fromiter((int(n) for n in G_und.nodes()), dtype=np.int64)
    if nodes.size == 0:
        return {}

    starts = nodes
    if num_starts < nodes.size:
        starts = rng.choice(nodes, size=int(num_starts), replace=False)

    nbrs = {int(n): np.fromiter((int(m) for m in G_und.neighbors(n)), dtype=np.int64) for n in G_und.nodes()}

    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    total = 0

    for s in starts:
        s = int(s)
        for _ in range(num_walks):
            prev = -1
            cur = s
            for _t in range(walk_len):
                neigh = nbrs.get(cur, None)
                if neigh is None or neigh.size == 0:
                    break

                if prev != -1 and neigh.size > 1:
                    candidates = neigh[neigh != prev]
                    if candidates.size == 0:
                        candidates = neigh
                else:
                    candidates = neigh

                nxt = int(candidates[rng.integers(0, candidates.size)])
                a, b = (cur, nxt) if cur < nxt else (nxt, cur)
                counts[(a, b)] += 1
                total += 1
                prev, cur = cur, nxt

    if total <= 0:
        return {(min(u, v), max(u, v)): 1.0 for (u, v) in G_und.edges()}

    eps = 1e-6
    wmap: Dict[Tuple[int, int], float] = {}
    for (u, v) in G_und.edges():
        a, b = (u, v) if u < v else (v, u)
        wmap[(int(a), int(b))] = float(counts.get((int(a), int(b)), 0)) + eps

    vals = np.array(list(wmap.values()), dtype=np.float64)
    mu = float(np.mean(vals)) if vals.size else 1.0
    if mu > 0:
        for k in list(wmap.keys()):
            wmap[k] = float(wmap[k] / mu)

    return wmap


def _cyclebreak_weights_transfer_shuffle(original_und_edges: np.ndarray,
                                        rewired_wmap: Dict[Tuple[int, int], float],
                                        seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w_rewired = np.array(list(rewired_wmap.values()), dtype=np.float32)
    if w_rewired.size == 0:
        return np.ones((original_und_edges.shape[0],), dtype=np.float32)

    rng.shuffle(w_rewired)
    if w_rewired.size < original_und_edges.shape[0]:
        idx = rng.integers(0, w_rewired.size, size=original_und_edges.shape[0])
        return w_rewired[idx]
    return w_rewired[: original_und_edges.shape[0]].copy()


def _cyclebreak_weights_transfer_degreepair(original_und_edges: np.ndarray,
                                           original_deg: np.ndarray,
                                           rewired_deg: np.ndarray,
                                           rewired_wmap: Dict[Tuple[int, int], float],
                                           seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    buckets: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    deg_values = set()

    for (a, b), w in rewired_wmap.items():
        da = int(rewired_deg[a])
        db = int(rewired_deg[b])
        p = (da, db) if da <= db else (db, da)
        buckets[p].append(float(w))
        deg_values.add(da)
        deg_values.add(db)

    if not buckets:
        return np.ones((original_und_edges.shape[0],), dtype=np.float32)

    deg_sorted = np.array(sorted(deg_values), dtype=np.int64)

    def nearest_deg(d: int) -> int:
        i = bisect_left(deg_sorted, d)
        if i == 0:
            return int(deg_sorted[0])
        if i >= deg_sorted.size:
            return int(deg_sorted[-1])
        lo = int(deg_sorted[i - 1])
        hi = int(deg_sorted[i])
        return lo if abs(d - lo) <= abs(d - hi) else hi

    out = np.empty((original_und_edges.shape[0],), dtype=np.float32)
    for i, (u, v) in enumerate(original_und_edges):
        du = int(original_deg[int(u)])
        dv = int(original_deg[int(v)])
        p = (du, dv) if du <= dv else (dv, du)

        choices = buckets.get(p, None)
        if not choices:
            ndu = nearest_deg(du)
            ndv = nearest_deg(dv)
            p2 = (ndu, ndv) if ndu <= ndv else (ndv, ndu)
            choices = buckets.get(p2, None)

        if not choices:
            any_key = next(iter(buckets.keys()))
            choices = buckets[any_key]

        out[i] = float(choices[rng.integers(0, len(choices))])

    mu = float(np.mean(out)) if out.size else 1.0
    if mu > 0:
        out = out / mu
    return out.astype(np.float32)


def _triangle_and_clustering(G_und: nx.Graph) -> Tuple[float, float]:
    try:
        cl = float(nx.average_clustering(G_und))
    except Exception:
        cl = float("nan")
    try:
        tri = nx.triangles(G_und)
        tcnt = float(sum(tri.values()) / 3.0)
    except Exception:
        tcnt = float("nan")
    return cl, tcnt


def build_variant_undirected_weights(variant: str,
                                    *,
                                    num_nodes: int,
                                    original_und_edges: np.ndarray,
                                    original_und_w: np.ndarray,
                                    seed: int) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(seed)
    diag: Dict = {}

    if variant == "full":
        return _weights_full(original_und_w), diag
    if variant == "uniform":
        return _weights_uniform(original_und_w), diag
    if variant == "shuffled":
        return _weights_shuffled(original_und_w, rng), diag
    if variant == "degree_only":
        w = _weights_degree_only(original_und_edges, num_nodes=int(num_nodes))
        return w, diag

    # cycle-broken variants
    swap_mult = int(os.environ.get("SWAP_MULT", str(SWAP_MULT_DEFAULT)))
    num_walks = int(os.environ.get("RNBRW_WALKS", str(RNBRW_WALKS_DEFAULT)))
    walk_len  = int(os.environ.get("RNBRW_WALKLEN", str(RNBRW_WALKLEN_DEFAULT)))
    num_starts = int(os.environ.get("RNBRW_STARTS", str(RNBRW_STARTS_DEFAULT)))

    G0 = _build_index_undirected_graph(int(num_nodes), original_und_edges)
    cl0, t0 = _triangle_and_clustering(G0)
    diag["orig_avg_clustering"] = cl0
    diag["orig_triangles"] = t0

    Grew = _rewire_degree_preserving(G0, seed=seed, swap_mult=swap_mult)
    cl1, t1 = _triangle_and_clustering(Grew)
    diag["rewired_avg_clustering"] = cl1
    diag["rewired_triangles"] = t1
    diag["swap_mult"] = int(swap_mult)
    diag["nswaps"] = int(swap_mult * Grew.number_of_edges())

    rew_wmap = _approx_rnbrw_weights(Grew, seed=seed, num_starts=num_starts, num_walks=num_walks, walk_len=walk_len)
    diag["rnbrw_num_starts"] = int(num_starts)
    diag["rnbrw_num_walks"] = int(num_walks)
    diag["rnbrw_walk_len"] = int(walk_len)
    diag["rewired_weight_count"] = int(len(rew_wmap))

    if variant == "cyclebreak_shuffle":
        w = _cyclebreak_weights_transfer_shuffle(original_und_edges, rew_wmap, seed=seed)
        return w, diag

    if variant == "cyclebreak_degreepair":
        orig_deg = np.zeros((int(num_nodes),), dtype=np.int64)
        for n, d in G0.degree():
            orig_deg[int(n)] = int(d)

        rew_deg = np.zeros((int(num_nodes),), dtype=np.int64)
        for n, d in Grew.degree():
            rew_deg[int(n)] = int(d)

        w = _cyclebreak_weights_transfer_degreepair(
            original_und_edges=original_und_edges,
            original_deg=orig_deg,
            rewired_deg=rew_deg,
            rewired_wmap=rew_wmap,
            seed=seed,
        )
        return w, diag

    raise ValueError(f"Unknown variant: {variant}")


# ------------------------
# Training + Evaluation (KMeans pipeline)
# ------------------------
def train_and_eval_variant(*,
                           x: torch.Tensor,
                           edge_index: torch.Tensor,
                           edge_weight_dir: torch.Tensor,
                           valid_nodes: np.ndarray,
                           gt_ge2: List[set],
                           seed: int,
                           device) -> Dict:
    lm = float(FIXED_CFG["lambda_mod"])
    ll = float(FIXED_CFG["lambda_lap"])
    lc = float(FIXED_CFG["lambda_contrast"])
    lo = float(FIXED_CFG["lambda_orth"])

    model = CAGATAblationModel(in_dim=x.size(1), hidden_dim=8, out_dim=int(K_FIXED), heads=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(EPOCHS):
        optim.zero_grad()
        _, _, embeddings = model(x, edge_index, edge_weight_dir)
        loss, _logs = combined_community_loss_PPI(
            embeddings=embeddings,
            edge_index=edge_index,
            edge_weight=edge_weight_dir,
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
        _, _, embeddings = model(x, edge_index, edge_weight_dir)

    Z = to_numpy(embeddings)         # [N, 793]
    Z_valid = Z[valid_nodes]         # [n_valid, 793]

    km = KMeans(n_clusters=int(K_FIXED), random_state=seed, n_init=10)
    km.fit(Z_valid)
    labels = km.labels_
    centroids = km.cluster_centers_  # [K, d]

    sil, ch, db = _safe_internal_metrics(Z_valid, labels)

    # top-r nearest centroids
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

    med, p90 = _community_size_stats(pred_ge2)

    if len(valid_nodes) < 2 or len(gt_ge2) == 0 or len(pred_ge2) == 0:
        onmi = np.nan
    else:
        onmi = float(onmi_mgh(gt_ge2, pred_ge2, nodes_mask=valid_nodes))

    return {
        "Seed": int(seed),
        "Variant": None,  # filled in main
        "K": int(K_FIXED),
        "r": int(R_FIXED),
        "Silhouette": (None if np.isnan(sil) else round(float(sil), 6)),
        "CH": (None if np.isnan(ch) else round(float(ch), 2)),
        "DB": (None if np.isnan(db) else round(float(db), 4)),
        "ONMI_r2": (None if np.isnan(onmi) else round(float(onmi), 6)),
        "PredComms_GE2_r2": int(len(pred_ge2)),
        "PredSize_Median_GE2_r2": (None if np.isnan(med) else round(float(med), 2)),
        "PredSize_P90_GE2_r2": (None if np.isnan(p90) else round(float(p90), 2)),
    }


# ------------------------
# Main
# ------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_idx, variant = _coalesce_args(seed_idx=None, variant=None)
    seed = SEEDS[int(seed_idx)]

    out_dir = _ensure_dirs()

    tag = variant.replace("-", "_")
    csv_path = os.path.join(out_dir, f"mechanism_{tag}_seed{seed}.csv")
    diag_path = os.path.join(out_dir, f"mechanism_{tag}_seed{seed}_diag.txt")
    error_path = os.path.join(out_dir, f"mechanism_{tag}_seed{seed}_error.txt")

    print(f"[RUN] PPI-CP RCN mechanism controls | seed={seed} variant={variant} | fixed K={K_FIXED} r={R_FIXED}")
    sys.stdout.flush()

    try:
        set_seed(seed)

        x, edge_index, edge_weight_dir_orig, num_nodes, und_edges, und_w, valid_nodes, gt_ge2 = _prepare_common(device)

        und_w_variant, diag = build_variant_undirected_weights(
            variant=variant,
            num_nodes=int(num_nodes),
            original_und_edges=und_edges,
            original_und_w=und_w,
            seed=seed,
        )

        # IMPORTANT baseline parity:
        # full -> use loader weights verbatim
        # others -> remap our undirected weights back to directed edge list
        if variant == "full":
            edge_weight_dir = edge_weight_dir_orig
        else:
            edge_weight_dir = _to_directed_weights(edge_index, und_edges, und_w_variant, device=device)

        with open(diag_path, "w") as f:
            f.write(f"seed={seed}\n")
            f.write(f"variant={variant}\n")
            f.write(f"K_fixed={K_FIXED}\n")
            f.write(f"r_fixed={R_FIXED}\n")
            for k, v in diag.items():
                f.write(f"{k}={v}\n")

        row = train_and_eval_variant(
            x=x,
            edge_index=edge_index,
            edge_weight_dir=edge_weight_dir,
            valid_nodes=valid_nodes,
            gt_ge2=gt_ge2,
            seed=seed,
            device=device,
        )
        row["Variant"] = variant

        pd.DataFrame([row]).to_csv(csv_path, index=False)
        print(f"[OK] Wrote {csv_path}")
        print(f"[OK] Wrote {diag_path}")

    except Exception as e:
        err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        with open(error_path, "w") as f:
            f.write(f"[ERROR]\nseed={seed}\nvariant={variant}\n")
            f.write(err_msg)
        print(f"[FAIL] wrote error log at {error_path}")
        raise


if __name__ == "__main__":
    main()
    os._exit(0)
