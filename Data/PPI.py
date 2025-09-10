import torch
import pandas as pd
import os
import pickle

def load_ppi_graph(use_onehot=False, device="cpu"):
    """
    Load the PPI graph and CORUM labels with overlapping communities.

    Returns:
        x, edge_index, y, edge_weight, G,
        communities, y_multi, mask, node_id_to_idx, idx_to_node_id
    """

    # --- Hard-coded paths (adjust to your setup) ---
    base_dir = os.path.dirname(__file__)  # directory where PPI.py lives
    data_dir = os.path.join(base_dir, "Graphs")

    ppi_path = os.path.join(data_dir, "ppi_graph.gpickle")
    corum_path = os.path.join(data_dir, "corum_humanComplexes.txt")
    with open(ppi_path, "rb") as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    N = len(nodes)

    # Map node IDs to indices
    node_id_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node_id = {i: n for i, n in enumerate(nodes)}

    # --- Features (always one-hot, ignore use_onehot flag) ---
    x = torch.eye(N, dtype=torch.float)

    # --- Edges ---
    edges, weights = [], []
    for u, v, attrs in G.edges(data=True):
        w = attrs.get("weight", 1.0)
        edges.append((node_id_to_idx[u], node_id_to_idx[v]))
        weights.append(w)
        # ensure undirected
        edges.append((node_id_to_idx[v], node_id_to_idx[u]))
        weights.append(w)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # --- CORUM labels ---
    corum = pd.read_csv(corum_path, sep="\t")
    complexes = []
    for _, row in corum.iterrows():
        cid = row["complex_id"]
        try:
            uniprots = row["subunits_uniprot_id"].split(";")
        except Exception:
            continue
        mapped = [node_id_to_idx[u] for u in uniprots if u in node_id_to_idx]
        if len(mapped) > 0:
            complexes.append((cid, set(mapped)))

    # Deduplicate identical (ComplexID, node-set) pairs
    seen = set()
    communities = []
    for cid, members in complexes:
        key = (cid, tuple(sorted(members)))
        if key not in seen:
            seen.add(key)
            communities.append(members)

    # --- Build labels ---
    node_to_comms = {i: set() for i in range(N)}
    for j, comm in enumerate(communities):
        for n in comm:
            node_to_comms[n].add(j)

    # Multi-hot
    C = len(communities)
    y_multi = torch.zeros((N, C), dtype=torch.long)
    for i, comms in node_to_comms.items():
        for c in comms:
            y_multi[i, c] = 1

    # Single-label placeholder (1D tensor)
    y = torch.full((N,), -1, dtype=torch.long)
    for i, comms in node_to_comms.items():
        if len(comms) > 0:
            y[i] = min(comms)  # deterministic choice

    # Mask: valid nodes for ONMI
    mask = (y != -1)

    return (
        x.to(device),
        edge_index.to(device),
        y.to(device),
        edge_weight.to(device),
        G,
        communities,
        y_multi.to(device),
        mask.to(device),
        node_id_to_idx,
        idx_to_node_id,
    )
