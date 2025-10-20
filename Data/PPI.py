import torch
import pandas as pd
import os
import pickle
import numpy as np  # used only for small summaries; safe to keep

def load_ppi_graph(use_onehot=False, device="cpu"):
    """
    Load the PPI graph and CORUM labels with overlapping communities.

    Returns:
        x, edge_index, y, edge_weight, G,
        communities, y_multi, mask, node_id_to_idx, idx_to_node_id
    """

    # --- Paths (unchanged) ---
    base_dir = os.path.dirname(__file__)  # directory where this file lives
    data_dir = os.path.join(base_dir, "Graphs")
    ppi_path   = os.path.join(data_dir, "ppi_graph.gpickle")
    corum_path = os.path.join(data_dir, "corum_humanComplexes.txt")
    hgnc_path  = os.path.join(data_dir, "hgnc_complete_set.txt")  # new dependency for ID bridging

    # --- Load PPI graph (unchanged) ---
    with open(ppi_path, "rb") as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    N = len(nodes)

    # Map node IDs to indices (unchanged)
    node_id_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node_id = {i: n for i, n in enumerate(nodes)}

    # --- Features (always one-hot, ignore use_onehot flag) ---
    x = torch.eye(N, dtype=torch.float)

    # --- Edges (unchanged; preserve weights and undirected doubling) ---
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

    # ============================
    # LABELS: CORUM -> HGNC -> Entrez -> PPI (THIS IS THE ONLY LOGIC WE CHANGE)
    # ============================

    # Helpers for normalization
    def _norm_sym(s):
        if not isinstance(s, str): return None
        s = s.strip()
        return s.upper() if s else None

    def _norm_uid(u):
        if not isinstance(u, str): return None
        u = u.strip()
        if not u: return None
        return u.split("-", 1)[0]  # drop isoform suffix e.g. P12345-2 -> P12345

    # Build fast lookups of PPI node IDs as Entrez (string and int forms)
    # Your PPI nodes are Entrez-like; they may be int or str. Support both.
    ppi_entrez_to_idx = {}
    for n, idx in node_id_to_idx.items():
        # string form
        ppi_entrez_to_idx[str(n)] = idx
        # integer form if possible
        if isinstance(n, str) and n.isdigit():
            ppi_entrez_to_idx[n] = idx
        elif isinstance(n, int):
            ppi_entrez_to_idx[str(n)] = idx

    # --- Load HGNC bridge (UniProt/Symbol -> Entrez) ---
    if not os.path.exists(hgnc_path):
        raise FileNotFoundError(
            f"[HGNC] Missing '{hgnc_path}'. Please place 'hgnc_complete_set.txt' in Graphs/."
        )
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str)
    # Ensure expected columns exist
    for col in ["symbol", "alias_symbol", "entrez_id", "uniprot_ids", "status"]:
        if col not in hgnc.columns:
            hgnc[col] = ""

    # Optional: restrict to current approved, protein-coding genes to reduce ambiguity
    # (keeps mapping cleaner; change if you want broader inclusion)
    # If you prefer not to filter by status/type, you can comment this block out.
    # Keep this conservative to avoid ambiguity fan-out by default.
    # Note: HGNC has 'status' (Approved/Entry Withdrawn). Protein-coding classification
    # may be in 'locus_type' or 'locus_group' depending on HGNC version. We won't hard filter by locus_type.
    hgnc = hgnc[hgnc["status"].fillna("").str.contains("Approved", na=False)]

    # Build symbol->entrez and uniprot->entrez maps (STRICT: drop ambiguous multi-Entrez keys)
    from collections import defaultdict
    sym_to_entrez_all = defaultdict(set)
    uid_to_entrez_all = defaultdict(set)

    # Normalize relevant columns
    hgnc["symbol"] = hgnc["symbol"].map(_norm_sym)
    hgnc["alias_symbol"] = hgnc["alias_symbol"].fillna("").astype(str)
    hgnc["entrez_id"] = hgnc["entrez_id"].fillna("").astype(str)
    hgnc["uniprot_ids"] = hgnc["uniprot_ids"].fillna("").astype(str)

    for _, r in hgnc.iterrows():
        entrez = r["entrez_id"].strip()
        if not entrez:
            continue

        # official symbol
        s = r["symbol"]
        if s:
            sym_to_entrez_all[s].add(entrez)

        # alias symbols (comma-separated)
        if r["alias_symbol"]:
            for a in r["alias_symbol"].split(","):
                a_norm = _norm_sym(a)
                if a_norm:
                    sym_to_entrez_all[a_norm].add(entrez)

        # UniProt IDs (comma-separated)
        if r["uniprot_ids"]:
            for u in r["uniprot_ids"].split(","):
                u_norm = _norm_uid(u)
                if u_norm:
                    uid_to_entrez_all[u_norm].add(entrez)

    # STRICT de-ambiguous maps: keep only 1-to-1 keys
    sym_to_entrez = {k: next(iter(v))
                     for k, v in sym_to_entrez_all.items() if len(v) == 1}
    uid_to_entrez = {k: next(iter(v))
                     for k, v in uid_to_entrez_all.items() if len(v) == 1}

    print(f"[DEBUG] HGNC sym_to_entrez size: {len(sym_to_entrez)}, uid_to_entrez size: {len(uid_to_entrez)}")
    print(f"[DEBUG] PPI sample node IDs (first 5): {list(node_id_to_idx.keys())[:5]}")


    # --- Load CORUM and normalize ---
    corum = pd.read_csv(corum_path, sep="\t", dtype=str)
    corum.columns = [c.lower() for c in corum.columns]

    # Safety: enforce human-only if column present
    if "organism" in corum.columns:
        org = corum["organism"].fillna("").str.lower()
        corum = corum[org.str.contains("homo sapiens") | org.str.contains("human")]

    if "complex_id" not in corum.columns:
        raise RuntimeError("[CORUM] Missing 'complex_id' column.")
    # We’ll use both subunits columns if available
    sym_col = "subunits_gene_name" if "subunits_gene_name" in corum.columns else None
    uid_col = "subunits_uniprot_id" if "subunits_uniprot_id" in corum.columns else None
    if sym_col is None and uid_col is None:
        raise RuntimeError("[CORUM] Need 'subunits_gene_name' and/or 'subunits_uniprot_id' columns.")

    # --- Map CORUM members -> Entrez (STRICT per-member; UNION across id types) ---
    # Policy:
    #   - For each token (symbol or UniProt), map via HGNC.
    #   - If a token maps to multiple Entrez (we already filtered those keys out), we’d drop it.
    #   - We UNION symbols + uniprot results for the complex member set.
    #   - Finally, intersect with PPI nodes and keep complexes with >=2 present members.

    complexes = []
    mapped_member_counts = []

    for _, row in corum.iterrows():
        cid = row["complex_id"]
        present_entrez = set()

        # Symbols
        if sym_col:
            raw = str(row.get(sym_col) or "")
            for s in [x for x in raw.split(";") if x]:
                s_norm = _norm_sym(s)
                if not s_norm:
                    continue
                e = sym_to_entrez.get(s_norm)  # strict: only 1-to-1 keys exist here
                if e and e in ppi_entrez_to_idx:
                    present_entrez.add(e)

        # UniProt
        if uid_col:
            raw = str(row.get(uid_col) or "")
            for u in [x for x in raw.split(";") if x]:
                u_norm = _norm_uid(u)
                if not u_norm:
                    continue
                e = uid_to_entrez.get(u_norm)  # strict: only 1-to-1 keys exist here
                if e and e in ppi_entrez_to_idx:
                    present_entrez.add(e)

        # Keep complex only if >=2 present-in-PPI members
        if len(present_entrez) >= 2:
            # Convert to node indices
            member_idxs = {ppi_entrez_to_idx[e] for e in present_entrez if e in ppi_entrez_to_idx}
            if len(member_idxs) >= 2:
                complexes.append((cid, member_idxs))
                mapped_member_counts.append(len(member_idxs))

    print(f"[DEBUG] Total complexes scanned: {len(corum)}")
    print(f"[DEBUG] Complexes retained (>=2 present): {len(complexes)}")
    print(f"[DEBUG] Example mapped complex (first 1): {complexes[0] if complexes else None}")


    # Optional small summary (safe)
    # (You can comment these prints out if you want zero console noise)
    # print(f"[CORUM] Complexes retained (>=2 present): {len(complexes)}")
    # if mapped_member_counts:
    #     print(f"[CORUM] Present-in-PPI members per kept complex: "
    #           f"mean={float(np.mean(mapped_member_counts)):.2f}, "
    #           f"min={int(np.min(mapped_member_counts))}, max={int(np.max(mapped_member_counts))}")

    # Deduplicate identical (complex_id, node-set) pairs by node-set only (keep one) — same spirit as your original
    seen = set()
    communities = []
    for cid, members in complexes:
        key = tuple(sorted(members))
        if key not in seen:
            seen.add(key)
            communities.append(set(members))

    # --- Build labels (unchanged interface; overlapping) ---
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

    # Single-label placeholder (1D tensor), deterministic if multiple
    y = torch.full((N,), -1, dtype=torch.long)
    for i, comms in node_to_comms.items():
        if len(comms) > 0:
            y[i] = min(comms)

    # Mask: nodes with any label
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


if __name__ == "__main__":
    # Run the loader
    (
        x, edge_index, y, edge_weight,
        G, communities, y_multi, mask,
        node_id_to_idx, idx_to_node_id
    ) = load_ppi_graph()

    print("\n=== LOADER TEST ===")
    print(f"Nodes: {x.shape[0]}, Features: {x.shape[1]}")
    print(f"Edges (directed count): {edge_index.shape[1]}")
    print(f"Edge weights: {edge_weight.shape}")
    print(f"Communities: {len(communities)}")
    print(f"Labeled nodes: {mask.sum().item()} / {mask.shape[0]} ({100*mask.sum().item()/mask.shape[0]:.2f}%)")

    # --- Edge sanity ---
    assert edge_index.shape[0] == 2, "edge_index must be shape [2, num_edges]"
    assert edge_index.shape[1] == edge_weight.shape[0], "edge_index and edge_weight length mismatch"
    assert torch.all(edge_weight > 0), "All edge weights should be > 0"
    print("Edge check: PASSED")

    # --- Label sanity ---
    N, C = y_multi.shape
    assert N == x.shape[0] == mask.shape[0], "Node dimension mismatch"
    assert y.shape[0] == N, "y length mismatch"
    assert (y_multi.sum(1) >= 0).all(), "Invalid y_multi entries"
    # Mask should match multi-hot
    assert torch.equal(mask, (y_multi.sum(1) > 0)), "Mask not aligned with y_multi"
    print("Label check: PASSED")

    # --- Spot check some edges and weights
    u, v = edge_index[:, 0].tolist()
    w = edge_weight[0].item()
    print(f"Sample edge: {u} -> {v}, weight={w}")

    # --- Spot check a node with labels
    labeled_nodes = torch.nonzero(mask, as_tuple=True)[0]
    if len(labeled_nodes) > 0:
        i = labeled_nodes[0].item()
        comms = torch.nonzero(y_multi[i]).flatten().tolist()
        print(f"Sample labeled node {i} has communities {comms}")
    else:
        print("No labeled nodes found!")

    print("=== LOADER TEST COMPLETE ===")
