import os, glob
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# ---------------- experiments/GAT/HelperFunctions.py ----------------
# Add this ONMI (MGH) implementation to your existing HelperFunctions module.

import math
import numpy as np
from typing import Iterable, List, Optional, Set

def onmi_mgh(
    communities_true: List[Set[int]],
    communities_pred: List[Set[int]],
    nodes_mask: Optional[Iterable[int]] = None,
) -> float:
    """
    Overlapping Normalized Mutual Information (McDaid–Greene–Hurley).

    Parameters
    ----------
    communities_true : list of sets
        Ground-truth overlapping communities as sets of node indices.
    communities_pred : list of sets
        Predicted (can be disjoint) communities as sets of node indices.
    nodes_mask : iterable of indices, optional
        Subset of node indices to include (e.g., unmasked nodes).
        If None, uses the union of all nodes present.

    Returns
    -------
    float
        ONMI in [0,1]; np.nan if undefined (degenerate/empty cases).
    """
    # --- Establish the universe of nodes to consider ---
    if nodes_mask is not None:
        U = set(nodes_mask)
    else:
        U = set()
        for c in communities_true:
            U |= set(c)
        for c in communities_pred:
            U |= set(c)

    # Drop empty and restrict communities to U
    X = []
    for c in communities_true:
        cc = set(c) & U
        if len(cc) > 0:
            X.append(cc)
    Y = []
    for c in communities_pred:
        cc = set(c) & U
        if len(cc) > 0:
            Y.append(cc)

    n = len(U)
    if n < 2 or len(X) == 0 or len(Y) == 0:
        return float("nan")

    # Optionally deduplicate identical node-sets (safe for GT; pred duplicates are rare)
    def _dedup(covers: List[Set[int]]) -> List[Set[int]]:
        seen = set()
        out = []
        for s in covers:
            key = frozenset(s)
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    X = _dedup(X)
    Y = _dedup(Y)

    if len(X) == 0 or len(Y) == 0:
        return float("nan")

    # Binary entropy helper (natural logs; base cancels in normalization)
    def h(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    # Entropy of a cover = mean entropy of its binary membership variables
    HX = np.mean([h(len(x) / n) for x in X])
    HY = np.mean([h(len(y) / n) for y in Y])

    if HX == 0.0 and HY == 0.0:
        # Both covers have zero entropy (all communities are trivial w.r.t U)
        return 1.0  # identical trivially; safe default
    # If only one side is zero-entropy, NMI is undefined for our normalization
    if HX == 0.0 or HY == 0.0:
        return float("nan")

    # Conditional entropy of Xi given Yj (weighted by P(Y))
    def H_cond(x: Set[int], y: Set[int]) -> float:
        a = len(x)                 # |X=1|
        b = len(y)                 # |Y=1|
        n11 = len(x & y)
        n10 = a - n11
        # n01 = b - n11
        # n00 = n - n11 - n10 - n01
        # P(Y=1), P(Y=0)
        p1 = b / n
        p0 = 1.0 - p1

        # H(X | Y=1)
        hx_y1 = h(n11 / b) if b > 0 else 0.0
        # H(X | Y=0)
        denom0 = n - b
        hx_y0 = h(n10 / denom0) if denom0 > 0 else 0.0

        return p1 * hx_y1 + p0 * hx_y0

    # Following MGH, allow matching against Y or its complement (take minimum)
    def H_cond_best(x: Set[int], y: Set[int]) -> float:
        h_xy = H_cond(x, y)
        # complement of y within U:
        ybar = U - y
        h_x_ybar = H_cond(x, ybar)
        return min(h_xy, h_x_ybar)

    # Mean conditional entropies using best matches
    HX_given_Y = np.mean([min(H_cond_best(x, y) for y in Y) for x in X])
    HY_given_X = np.mean([min(H_cond_best(y, x) for x in X) for y in Y])

    # Normalized form
    nmi = 1.0 - 0.5 * ((HX_given_Y / HX) + (HY_given_X / HY))
    # Clamp tiny numerical drift
    if nmi < 0 and nmi > -1e-12:
        nmi = 0.0
    if nmi > 1 and nmi < 1 + 1e-12:
        nmi = 1.0
    return float(nmi)


def elbow_method(embeddings, max_k=10, model_name=None):
    if hasattr(embeddings, 'detach'):
        embeddings = embeddings.detach().cpu().numpy()

    inertias = []
    ks = list(range(2, max_k + 1))

    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    if model_name:  # Save instead of show
        plt.figure(figsize=(6, 5))
        plt.plot(ks, inertias, 'o-')
        plt.title(f"Elbow Method: {model_name}")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"elbow_{model_name}.png", dpi=300)
        plt.close()

    return ks, inertias

def plot_training_summaries(dataset: str, base_dir: str = "logs", final_subdir: str = "FinalResults"):
    """
    Produces high-res, publication-ready plots:
      - Loss_vs_Epoch.png  (legend upper-right)
      - Silhouette_vs_Epoch.png (legend lower-left)
    Saved to logs/{dataset}/FinalResults/
    """
    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def _safe_float_series(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)

    def _collect_mean_curves_for_model(model_dir: str):
        seed_csvs = sorted(glob.glob(os.path.join(model_dir, "seed_*.csv")))
        if not seed_csvs:
            return None, None, None, 0
        loss_list, sil_list, lengths = [], [], []
        for path in seed_csvs:
            try:
                df = pd.read_csv(path)
                if "Loss" not in df.columns or "Silhouette" not in df.columns:
                    continue
                loss = _safe_float_series(df["Loss"]).to_numpy()
                sil  = _safe_float_series(df["Silhouette"]).to_numpy()
                mask = np.isfinite(loss) & np.isfinite(sil)
                loss, sil = loss[mask], sil[mask]
                if len(loss) == 0 or len(sil) == 0:
                    continue
                lengths.append(min(len(loss), len(sil)))
                loss_list.append(loss)
                sil_list.append(sil)
            except Exception:
                continue
        if not lengths:
            return None, None, None, 0
        min_len = min(lengths)
        loss_trim = np.stack([arr[:min_len] for arr in loss_list], axis=0)
        sil_trim  = np.stack([arr[:min_len] for arr in sil_list],  axis=0)
        mean_loss = np.nanmean(loss_trim, axis=0)
        mean_sil  = np.nanmean(sil_trim,  axis=0)
        epochs = np.arange(1, min_len + 1, dtype=int)
        return epochs, mean_loss, mean_sil, loss_trim.shape[0]

    root = os.path.join(base_dir, dataset)
    out_dir = os.path.join(root, final_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Discover models (skip FinalResults)
    model_dirs = [d for d in sorted(glob.glob(os.path.join(root, "*")))
                  if os.path.isdir(d) and os.path.basename(d) != final_subdir]

    curves = {}
    for d in model_dirs:
        model = os.path.basename(d)
        ep, m_loss, m_sil, n = _collect_mean_curves_for_model(d)
        if ep is None or n == 0:
            continue
        curves[model] = (ep, m_loss, m_sil, n)

    if not curves:
        print(f"[WARN] No per-epoch seed CSVs found under {root}. Skipping plots.")
        return

    # --- Order & styling: put RCN first, then alphabetical for the rest ---
    RCN_NAME = "RCN"
    model_order = sorted(curves.keys(), key=lambda m: (m != RCN_NAME, m.lower()))

    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i % 10) for i, m in enumerate(model_order)}

    rc = {
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 14, "axes.titlesize": 20, "axes.labelsize": 16, "legend.fontsize": 12,
    }

    # 1) Loss vs Epoch (legend upper-right)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(12.8, 7.68))
        for m in model_order:
            ep, ml, ms, n = curves[m]
            lw = 2.8 if m == RCN_NAME else 2.0
            ax.plot(ep, ml, label=m, lw=lw, color=color_map[m])
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"{dataset}: Loss vs Epoch"); ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Loss_vs_Epoch.png"))
        plt.close(fig)

    # 2) Silhouette vs Epoch (legend lower-left)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(12.8, 7.68))
        for m in model_order:
            ep, ml, ms, n = curves[m]
            lw = 2.8 if m == RCN_NAME else 2.0
            ax.plot(ep, ms, label=m, lw=lw, color=color_map[m])
        ax.set_xlabel("Epoch"); ax.set_ylabel("Silhouette Score")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{dataset}: Silhouette vs Epoch"); ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Silhouette_vs_Epoch.png"))
        plt.close(fig)

    print(f"[OK] Saved plots to {out_dir}")

def build_metrics_table(dataset: str, base_dir: str = "logs", final_subdir: str = "FinalResults"):
    """
    Reads logs/{dataset}/FinalResults/{dataset}_Results.csv, computes per-model
    mean±std for ARI, NMI, Silhouette, and saves metrics_table.csv.
    """
    out_dir = os.path.join(base_dir, dataset, final_subdir)
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, f"{dataset}_Results.csv")
    if not os.path.exists(results_csv):
        print(f"[WARN] Results not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    if "Dataset" in df.columns:
        df = df[df["Dataset"].astype(str) == str(dataset)]

    # Normalize the Silhouette column name
    sil_cols = [c for c in df.columns if c.lower().startswith("silhouette")]
    sil_col = sil_cols[0] if sil_cols else None

    for c in filter(None, ["ARI", "NMI", sil_col]):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grp = df.groupby("Model", dropna=False).agg(
        ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"),
        NMI_mean=("NMI", "mean"), NMI_std=("NMI", "std"),
        Sil_mean=(sil_col, "mean") if sil_col else ("NMI", "mean"),
        Sil_std =(sil_col, "std")  if sil_col else ("NMI", "std"),
        runs=("Model", "count"),
    ).reset_index()

    def pm(m, s):
        if pd.isna(m): return ""
        if pd.isna(s): s = 0.0
        return f"{m:.4f} ± {s:.4f}"

    out = pd.DataFrame({
        "Model": grp["Model"],
        "Runs": grp["runs"],
        "ARI_mean": grp["ARI_mean"].round(4),
        "ARI_std":  grp["ARI_std"].fillna(0).round(4),
        "ARI (mean ± std)": [pm(m, s) for m, s in zip(grp["ARI_mean"], grp["ARI_std"])],
        "NMI_mean": grp["NMI_mean"].round(4),
        "NMI_std":  grp["NMI_std"].fillna(0).round(4),
        "NMI (mean ± std)": [pm(m, s) for m, s in zip(grp["NMI_mean"], grp["NMI_std"])],
        "Silhouette_mean": grp["Sil_mean"].round(4),
        "Silhouette_std":  grp["Sil_std"].fillna(0).round(4),
        "Silhouette (mean ± std)": [pm(m, s) for m, s in zip(grp["Sil_mean"], grp["Sil_std"])],
    })

    out_path = os.path.join(out_dir, "metrics_table.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] Saved metrics table to: {out_path}")

def generate_final_results(dataset: str, base_dir: str = "logs", final_subdir: str = "FinalResults"):
    """
    Call this after saving {base_dir}/{dataset}/{final_subdir}/{dataset}_Results.csv
    """
    plot_training_summaries(dataset, base_dir=base_dir, final_subdir=final_subdir)
    build_metrics_table(dataset, base_dir=base_dir, final_subdir=final_subdir)

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def plot_embedding_2d(
    embeddings,
    title="Embeddings",
    method="pca",            # 'pca' | 'tsne' | 'umap'
    labels=None,             # optional color labels (e.g., y_true or cluster ids)
    k=None,                  # if set, run KMeans on FULL-D embeddings and color by its labels
    random_state=42,
    tsne_perplexity=30,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    annotate=True
):
    """
    - Reduces to 2D with PCA/t-SNE/UMAP for plotting.
    - If k is provided AND labels is None, runs KMeans on full-D embeddings to color points.
    - If labels is provided, uses them for coloring (overrides k).
    """
    X = _to_numpy(embeddings)

    # Optional clustering in the ORIGINAL space (not the 2D projection)
    if labels is None and k is not None:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)

    # 2D reduction
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        X2 = reducer.fit_transform(X)
        method_name = "PCA"
    elif method == "tsne":
        reducer = TSNE(
            n_components=2, perplexity=tsne_perplexity, init="pca",
            learning_rate="auto", random_state=random_state
        )
        X2 = reducer.fit_transform(X)
        method_name = f"t-SNE"
    else:
        raise ValueError("method must be 'pca', 'tsne', or 'umap'")

    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(X2[:, 0], X2[:, 1], s=100, c="skyblue", edgecolor="k")
    else:
        # discrete colormap for clusters/labels
        num_classes = int(np.max(labels)) + 1
        cmap = plt.cm.get_cmap("tab20", num_classes)
        plt.scatter(X2[:, 0], X2[:, 1], s=100, c=labels, cmap=cmap, edgecolor="k")

    if annotate:
        for i, (x, y) in enumerate(X2):
            plt.text(x, y, str(i), fontsize=9, ha="center", va="center")

    t = f"{title} – ({method_name})"
    plt.title(t)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_elbow_and_silhouette(
    embeddings,
    k_range=range(2, 16),
    random_state=42
):
    """
    Runs KMeans in FULL-D space and plots:
      - Inertia (elbow)
      - Silhouette score
    Returns (ks, inertias, silhouettes).
    """
    X = _to_numpy(embeddings)
    inertias, sils = [], []
    ks = list(k_range)

    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # Silhouette only defined for k>1 and < n_samples
        if len(np.unique(labels)) > 1 and k < len(X):
            sils.append(silhouette_score(X, labels))
        else:
            sils.append(np.nan)

    # Inertia plot
    plt.figure(figsize=(6, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Silhouette plot
    plt.figure(figsize=(6, 5))
    plt.plot(ks, sils, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette vs k (KMeans)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return ks, inertias, sils

def plot_cosine_similarity_heatmap(embeddings, assignments, title="Cosine Similarity Heatmap"):
    """
    Plots a cosine similarity heatmap of embeddings using community-based ordering and formatting
    similar to `plot_heatmap`.

    Args:
        embeddings (torch.Tensor): Node embeddings of shape [N, D]
        assignments (torch.Tensor): Cluster labels of shape [N]
        title (str): Title of the plot
    """
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    if assignments.is_cuda:
        assignments = assignments.cpu()
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()

    embeddings = F.normalize(embeddings, dim=1)

    # Create partition and sort node order by community label
    partition = {}
    for idx, label in enumerate(assignments):
        partition[idx] = label.item() if torch.is_tensor(label) else label
    order = sorted(partition.keys(), key=lambda x: partition[x])

    # Build cosine similarity matrix
    N = len(order)
    idx_map = {node: i for i, node in enumerate(order)}
    sim_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            u, v = order[i], order[j]
            sim = F.cosine_similarity(embeddings[u].unsqueeze(0), embeddings[v].unsqueeze(0), dim=1).item()
            sim_matrix[i, j] = sim

    # Identify community boundaries
    boundaries = []
    last_comm = partition[order[0]]
    for i, node in enumerate(order[1:], start=1):
        comm = partition[node]
        if comm != last_comm:
            boundaries.append(i - 0.5)
            last_comm = comm

    # Plot using imshow (like your style)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sim_matrix, aspect='auto', cmap='coolwarm', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title(title)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticklabels(order, fontsize=6)

    for b in boundaries:
        ax.axhline(b, color='red', linewidth=1.5)
        ax.axvline(b, color='red', linewidth=1.5)

    plt.tight_layout()
    plt.show()



def save_rnbrw_modularity_results(assignments, G, out_csv_path):
    """
    Uses existing NetworkX graph G with RNBRW weights to compute modularity,
    and saves the community assignments.

    Parameters:
    - assignments: Tensor or array of cluster labels [N]
    - G: NetworkX graph with RNBRW edge weights stored as 'weight'
    - out_csv_path: Output path for CSV file with columns: Node, Community
    """

    if hasattr(assignments, 'cpu'):
        assignments = assignments.cpu().numpy()

    # Group nodes into communities
    partition = {}
    for node, label in enumerate(assignments):
        partition.setdefault(label, []).append(node)
    communities = list(partition.values())

    # Compute RNBRW-weighted modularity directly from G
    modularity_score = nx.algorithms.community.modularity(G, communities, weight='weight')

    # Save community assignments
    df = pd.DataFrame({
        'Node': np.arange(len(assignments)),
        'Community': assignments
    })
    df.to_csv(out_csv_path, index=False)

    print(f"✅ Community CSV saved to {out_csv_path}")
    print(f"📊 RNBRW-weighted modularity: {modularity_score:.5f}")
    return modularity_score

def cluster_with_kmeans(embeddings: torch.Tensor, num_clusters: int, seed: int = 42) -> torch.Tensor:
    """
    Applies K-means clustering to node embeddings.

    Args:
        embeddings (torch.Tensor): Node embeddings of shape [N, D]
        num_clusters (int): Number of clusters to find (e.g., number of communities)
        seed (int): Random seed for reproducibility

    Returns:
        torch.Tensor: Cluster labels as a tensor of shape [N]
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')
    labels = kmeans.fit_predict(embeddings_np)
    return torch.tensor(labels, dtype=torch.long)

def plot_parameter_sensitivity_heatmaps(results, temperatures, head_counts, metric_name="Accuracy"):
    """
    Plots a heatmap for each attention strategy, showing performance over
    (temperature × number of heads) combinations.

    Args:
        results (dict): Nested dictionary from ablation loop:
                        {strategy: {temperature: {heads: value}}}
        temperatures (list): List of temperature values (columns)
        head_counts (list): List of attention head counts (rows)
        metric_name (str): Label for the performance metric (default: "Accuracy")
    """
    for strategy, strat_results in results.items():
        heatmap_matrix = np.zeros((len(head_counts), len(temperatures)))

        for i, heads in enumerate(head_counts):
            for j, temp in enumerate(temperatures):
                value = strat_results.get(temp, {}).get(heads, np.nan)
                heatmap_matrix[i, j] = value

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_matrix,
            xticklabels=temperatures,
            yticklabels=head_counts,
            annot=True,
            cmap="viridis",
            fmt=".3f"
        )
        plt.title(f"{strategy} – {metric_name} Sensitivity")
        plt.xlabel("Temperature")
        plt.ylabel("Number of Heads")
        plt.tight_layout()
        plt.show()

def plot_heatmap(edge_u, edge_v, weights, partition, order, title):
    G = nx.Graph()
    G.add_nodes_from(order)  # use supplied order!
    for u, v, w in zip(edge_u, edge_v, weights):
        G.add_edge(u, v, weight=w)

    idx = {n: i for i, n in enumerate(order)}
    N = len(order)

    A = np.zeros((N, N))
    for u, v, w in zip(edge_u, edge_v, weights):
        i, j = idx[u], idx[v]
        A[i, j] = A[j, i] = w

    boundaries = []
    last_comm = partition[order[0]]
    for i, node in enumerate(order[1:], start=1):
        comm = partition[node]
        if comm != last_comm:
            boundaries.append(i - 0.5)
            last_comm = comm

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(A, aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label="Weight")
    ax.set_title(title)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticklabels(order, fontsize=6)
    for b in boundaries:
        ax.axhline(b, color='red', linewidth=1.5)
        ax.axvline(b, color='red', linewidth=1.5)


def find_cycles_dfs(G, max_len=8):
    """
    Finds all simple cycles in an undirected graph up to a maximum length.
    Returns a list of cycles (each is a list of node IDs).
    """
    def dfs(curr, start, visited, path, cycles):
        if len(path) > max_len:
            return
        visited.add(curr)
        path.append(curr)

        for neighbor in G[curr]:
            if neighbor == start and len(path) > 2:
                cycles.add(tuple(sorted(path)))
            elif neighbor not in visited:
                dfs(neighbor, start, visited, path, cycles)

        path.pop()
        visited.remove(curr)

    all_cycles = set()
    for node in G.nodes():
        dfs(node, node, set(), [], all_cycles)

    return [list(cycle) for cycle in all_cycles]


def find_cycles_by_starting_club(G, cycles):
    blue_count = 0
    orange_count = 0
    seen = set()

    for cycle in cycles:
        start_node = cycle[0]
        normalized = tuple(sorted(cycle))
        if normalized not in seen:
            seen.add(normalized)
            club = G.nodes[start_node]["club"]
            if club == "Mr. Hi":
                blue_count += 1
            elif club == "Officer":
                orange_count += 1

    return blue_count, orange_count

def count_edge_in_cycles(edge, cycles):
    u, v = edge
    count = 0
    for cycle in cycles:
        cycle_edges = set()
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            cycle_edges.add(frozenset((a, b)))
        if frozenset((u, v)) in cycle_edges:
            count += 1
    return count

def get_topk_neighbor_mask(H, edge_index, edge_weight=None, use_rnbrw=True):
    """
    Precompute top-k neighbor mask using the rule:
    k_i = min(max(5, floor(degree_i / 2)), 20)
    Applies to both RNBRW and adjacency-based methods.

    Args:
        H: Node embeddings (N x D)
        edge_index: Edge index tensor (2 x E)
        edge_weight: RNBRW weights if applicable
        use_rnbrw: Whether to use RNBRW weights (True) or unweighted adjacency (False)

    Returns:
        topk_mask: [N x N] BoolTensor indicating top-k neighbors for each node
    """
    N, D = H.size()
    device = H.device

    # Build adjacency matrix
    adj = torch.zeros((N, N), device=device)
    src, dst = edge_index

    if use_rnbrw and edge_weight is not None:
        adj[src, dst] = edge_weight
        adj[dst, src] = edge_weight
    else:
        adj[src, dst] = 1
        adj[dst, src] = 1

    adj.fill_diagonal_(0)
    degree = (adj > 0).sum(dim=1)  # [N]
    num_pos = torch.clamp((degree.float() / 2).floor(), min=5, max=20).long()  # [N]

    topk_mask = torch.zeros_like(adj, dtype=torch.bool)

    for i in range(N):
        if degree[i] == 0:
            continue

        if use_rnbrw:
            sorted_indices = torch.argsort(adj[i], descending=True)
            top_ids = sorted_indices[:num_pos[i]]
        else:
            neighbors = (adj[i] > 0).nonzero(as_tuple=True)[0]
            if neighbors.numel() == 0:
                continue
            sims = F.cosine_similarity(H[i].unsqueeze(0), H[neighbors], dim=-1)
            top_ids = neighbors[sims.topk(min(num_pos[i], neighbors.numel())).indices]

        topk_mask[i, top_ids] = True

    return topk_mask