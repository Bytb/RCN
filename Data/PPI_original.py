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
    
    
    
    # --- Build alias maps from node attributes, if present ---
    symbol_to_idx = {}
    uniprot_to_idx = {}
    
    for n in nodes:
        attrs = G.nodes[n] if hasattr(G, "nodes") else {}
        # common keys people store
        sym = attrs.get("symbol") or attrs.get("gene_symbol") or attrs.get("GENE_SYMBOL")
        up  = attrs.get("uniprot") or attrs.get("uniprot_id") or attrs.get("UNIPROT")
    
        if isinstance(sym, str) and len(sym.strip()) > 0:
            symbol_to_idx[sym.strip().upper()] = node_id_to_idx[n]
    
        if isinstance(up, str) and len(up.strip()) > 0:
            # normalize UniProt (drop isoform suffix like P12345-2)
            up_norm = up.strip().split("-", 1)[0]
            uniprot_to_idx[up_norm] = node_id_to_idx[n]
    
    def norm_uid(u: str):
        if not isinstance(u, str): return None
        u = u.strip()
        if not u: return None
        return u.split("-", 1)[0]  # drop isoform
    
    def norm_sym(s: str):
        if not isinstance(s, str): return None
        s = s.strip().upper()
        return s if s else None

    
        # ---- CORUM sanity: columns & first rows ----
    print(f"[CORUM] Columns: {list(corum.columns)}")
    print(corum.head(3).to_string())
    
    # Pick the right columns (adjust if your file differs)
    id_col = "complex_id"
    members_col = "subunits_uniprot_id"
    if id_col not in corum.columns or members_col not in corum.columns:
        raise RuntimeError("[CORUM] Expected columns not found. "
                           f"Have: {list(corum.columns)}; need '{id_col}', '{members_col}'.")
    
    # Optional: filter to human if your file mixes species
    # if "Organism" in corum.columns:
    #     corum = corum[corum["Organism"].str.contains("Homo sapiens", na=False)]
    
    # Normalize UniProt IDs: drop isoform suffixes and whitespace
    def norm_uid(u):
        if not isinstance(u, str):
            return None
        u = u.strip()
        if u == "":
            return None
        # remove isoform suffix "-1", "-2", etc.
        if "-" in u:
            base = u.split("-", 1)[0]
        else:
            base = u
        return base
    
    # How many node IDs exist and an example
    print(f"[PPI] Node ID sample (first 5): {list(node_id_to_idx.keys())[:5]}")
    N_before = len(node_id_to_idx)
    
    complexes = []
    mapped_member_counts = []
    
    for _, row in corum.iterrows():
        try:
            members_raw = str(row[members_col]).split(";")
        except Exception:
            continue
        members = [norm_uid(u) for u in members_raw]
        members = [u for u in members if u is not None]
        mapped = [node_id_to_idx[u] for u in members if u in node_id_to_idx]
        if len(mapped) > 0:
            complexes.append((row[id_col], set(mapped)))
            mapped_member_counts.append(len(mapped))
    
    print(f"[CORUM] Complexes parsed: {len(complexes)}")
    if len(mapped_member_counts) > 0:
        print(f"[CORUM] Mapped members per complex: "
              f"mean={float(np.mean(mapped_member_counts)):.2f}, "
              f"min={int(np.min(mapped_member_counts))}, max={int(np.max(mapped_member_counts))}")
    else:
        print("[CORUM] No complex members mapped onto PPI node IDs. "
              "Likely an ID-format mismatch (e.g., UniProt vs gene symbols) or wrong columns/file.")

    
    
    
    
    
    
    
    
    complexes = []
    mapped_member_counts = []
    
    for _, row in corum.iterrows():
        cid = row["complex_id"]
        # members as symbols (preferred, often easier to align)
        sym_raw = str(row.get("subunits_gene_name", "") or "")
        syms = [norm_sym(x) for x in sym_raw.split(";") if x != ""]
        # members as UniProt (fallback)
        up_raw = str(row.get("subunits_uniprot_id", "") or "")
        ups = [norm_uid(x) for x in up_raw.split(";") if x != ""]
    
        mapped = set()
    
        # Try symbols first
        for s in syms:
            if s is not None and s in symbol_to_idx:
                mapped.add(symbol_to_idx[s])
    
        # Fallback to UniProt if symbols didn’t map
        if len(mapped) == 0:
            for u in ups:
                if u is not None and u in uniprot_to_idx:
                    mapped.add(uniprot_to_idx[u])
    
        if len(mapped) > 0:
            complexes.append((cid, mapped))
            mapped_member_counts.append(len(mapped))
    
    print(f"[CORUM] Complexes parsed (with symbol/UniProt aliasing): {len(complexes)}")
    if len(mapped_member_counts) > 0:
        print(f"[CORUM] Mapped members per complex: "
              f"mean={float(np.mean(mapped_member_counts)):.2f}, "
              f"min={int(np.min(mapped_member_counts))}, max={int(np.max(mapped_member_counts))}")
    else:
        print("[CORUM] Still no members mapped. Likely your graph nodes are raw Entrez IDs without symbol/uniprot attributes.")
    

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
